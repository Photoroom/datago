package datago

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync"

	"github.com/davidbyttow/govips/v2/vips"
	_ "go.uber.org/automaxprocs/maxprocs"
)

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Public interface for this package, will be reflected in the python bindings

// All the supported source types
type DatagoSourceType string

const (
	SourceTypeDB         DatagoSourceType = "DB"
	SourceTypeFileSystem DatagoSourceType = "filesystem"
)

// Nested configuration structures for the client
type DataSourceConfig struct {
	PageSize  int `json:"page_size"`
	Rank      int `json:"rank"`
	WorldSize int `json:"world_size"`
	Limit     int `json:"limit"`
}

type ImageTransformConfig struct {
	CropAndResize     bool    `json:"crop_and_resize"`
	DefaultImageSize  int     `json:"default_image_size"`
	DownsamplingRatio int     `json:"downsampling_ratio"`
	MinAspectRatio    float64 `json:"min_aspect_ratio"`
	MaxAspectRatio    float64 `json:"max_aspect_ratio"`
	PreEncodeImages   bool    `json:"pre_encode_images"`
}

func (c *ImageTransformConfig) setDefaults() {
	c.DefaultImageSize = 512
	c.DownsamplingRatio = 16
	c.MinAspectRatio = 0.5
	c.MaxAspectRatio = 2.0
	c.PreEncodeImages = false
}

// DatagoConfig is the main configuration structure for the datago client
type DatagoConfig struct {
	SourceType         DatagoSourceType     `json:"source_type"`
	SourceConfig       interface{}          `json:"source_config"`
	ImageConfig        ImageTransformConfig `json:"image_config"`
	PrefetchBufferSize int                  `json:"prefetch_buffer_size"`
	SamplesBufferSize  int                  `json:"samples_buffer_size"`
	Concurrency        int                  `json:"concurrency"`
	Limit              int                  `json:"limit"`
}

func (c *DatagoConfig) setDefaults() {
	dbConfig := SourceDBConfig{}
	dbConfig.setDefaults()
	c.SourceConfig = dbConfig

	c.ImageConfig.setDefaults()
	c.PrefetchBufferSize = 64
	c.SamplesBufferSize = 32
	c.Concurrency = 64
	c.Limit = 0
}

func GetDatagoConfig() DatagoConfig {
	config := DatagoConfig{}
	config.setDefaults()
	return config
}

func DatagoConfigFromJSON(jsonString string) DatagoConfig {
	config := DatagoConfig{}
	var tempConfig map[string]interface{}
	err := json.Unmarshal([]byte(jsonString), &tempConfig)
	if err != nil {
		log.Panicf("Error unmarshalling JSON: %v", err)
	}

	sourceConfig, err := json.Marshal(tempConfig["source_config"])
	if err != nil {
		fmt.Println("Error marshalling source_config", tempConfig["source_config"], err)
		log.Panicf("Error marshalling source_config: %v", err)
	}

	// Unmarshal the source config based on the source type
	// NOTE: The undefined fields will follow the default values
	switch tempConfig["source_type"] {
	case string(SourceTypeDB):
		dbConfig := SourceDBConfig{}
		dbConfig.setDefaults()

		err = json.Unmarshal(sourceConfig, &dbConfig)
		if err != nil {
			fmt.Println("Error unmarshalling DB config", sourceConfig, err)
			log.Panicf("Error unmarshalling DB config: %v", err)
		}
		config.SourceConfig = dbConfig
	case string(SourceTypeFileSystem):
		fsConfig := SourceFileSystemConfig{}
		fsConfig.setDefaults()

		err = json.Unmarshal(sourceConfig, &fsConfig)
		if err != nil {
			fmt.Println("Error unmarshalling Filesystem config", sourceConfig, err)
			log.Panicf("Error unmarshalling FileSystem config: %v", err)
		}
		config.SourceConfig = fsConfig
	default:
		fmt.Println(tempConfig["source_type"] == SourceTypeFileSystem, tempConfig["source_type"], SourceTypeFileSystem)
		log.Panic("Unsupported source type ", tempConfig["source_type"])
	}

	imageConfig, err := json.Marshal(tempConfig["image_config"])
	if err != nil {
		log.Panicf("Error marshalling image_config: %v", err)
	}
	err = json.Unmarshal(imageConfig, &config.ImageConfig)
	if err != nil {
		log.Panicf("Error unmarshalling Image config: %v", err)
	}

	config.PrefetchBufferSize = int(tempConfig["prefetch_buffer_size"].(float64))
	config.SamplesBufferSize = int(tempConfig["samples_buffer_size"].(float64))
	config.Concurrency = int(tempConfig["concurrency"].(float64))
	if err != nil {
		log.Panicf("Error unmarshalling JSON: %v", err)
	}
	return config
}

// DatagoClient is the main client structure, will be instantiated by the user
type DatagoClient struct {
	context   context.Context
	waitGroup *sync.WaitGroup
	cancel    context.CancelFunc

	imageConfig   ImageTransformConfig
	servedSamples int
	limit         int

	// Flexible generator, backend and dispatch goroutines
	generator Generator
	backend   Backend

	// Channels	- these will be used to communicate between the background goroutines
	chanPages          chan Pages
	chanSampleMetadata chan SampleDataPointers
	chanSamples        chan Sample
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// GetClient is a constructor for the DatagoClient, given a JSON configuration string
func GetClient(config DatagoConfig) *DatagoClient {
	// Initialize the vips library
	vips.LoggingSettings(nil, vips.LogLevelWarning)
	vips.Startup(nil)

	// Create the generator and backend
	var generator Generator
	var backend Backend

	// Pedestrial, but we'll just switch over all the options for now
	// the source type will determine the generator and backend
	// the client will eventually expose the samples to the user
	fmt.Println(reflect.TypeOf(config.SourceConfig))

	switch config.SourceConfig.(type) {
	case SourceDBConfig:
		fmt.Println("Creating a DB-backed dataloader")
		dbConfig := config.SourceConfig.(SourceDBConfig)
		generator = newDatagoGeneratorDB(dbConfig)
		backend = BackendHTTP{config: &dbConfig, concurrency: config.Concurrency}
	case SourceFileSystemConfig:
		fmt.Println("Creating a FileSystem-backed dataloader")
		fsConfig := config.SourceConfig.(SourceFileSystemConfig)
		generator = newDatagoGeneratorFileSystem(fsConfig)
		backend = BackendFileSystem{config: &config, concurrency: config.Concurrency}
	default:
		fmt.Println("Unsupported source type")
		log.Panic("Unsupported source type")
	}

	// Create the client
	client := &DatagoClient{
		chanPages:          make(chan Pages, 2),
		chanSampleMetadata: make(chan SampleDataPointers, config.PrefetchBufferSize),
		chanSamples:        make(chan Sample, config.SamplesBufferSize),
		imageConfig:        config.ImageConfig,
		servedSamples:      0,
		limit:              config.Limit,
		context:            nil,
		cancel:             nil,
		waitGroup:          nil,
		generator:          generator,
		backend:            backend,
	}

	// Make sure that the client will be Stopped() upon destruction
	runtime.SetFinalizer(client, func(r *DatagoClient) {
		r.Stop()
	})

	return client
}

func GetClientFromJSON(jsonString string) *DatagoClient {
	config := DatagoConfigFromJSON(jsonString)
	return GetClient(config)
}

// Start the background downloads, make it ready to serve samples. Will grow the memory and CPU footprint
func (c *DatagoClient) Start() {
	if c.context == nil || c.cancel == nil {
		// Get a context and a cancel function to stop the background goroutines and gracefully handle
		// interruptions at during http round trips
		c.context, c.cancel = context.WithCancel(context.Background())
	}

	// Report panics in the background goroutines
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Caught a panic: %v", r)
			log.Printf("Stack trace: %s", debug.Stack())
			os.Exit(-1)
		}
	}()

	// Optionally crop and resize the images and masks on the fly
	var arAwareTransform *ARAwareTransform = nil

	if c.imageConfig.CropAndResize {
		fmt.Println("Cropping and resizing images")
		fmt.Println("Base image size | downsampling ratio | min | max:", c.imageConfig.DefaultImageSize, c.imageConfig.DownsamplingRatio, c.imageConfig.MinAspectRatio, c.imageConfig.MaxAspectRatio)
		arAwareTransform = newARAwareTransform(c.imageConfig)
	}

	if c.imageConfig.PreEncodeImages {
		fmt.Println("Pre-encoding images, we'll return serialized JPG and PNG bytes")
	}

	// Collect the pages, metadata and items in the background.
	var wg sync.WaitGroup

	// Start all goroutines and log them in a waitgroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		c.generator.generatePages(c.context, c.chanPages) // Collect the root data source pages
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.asyncDispatch() // Dispatch the content of the pages to the items channel
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.backend.collectSamples(c.chanSampleMetadata, c.chanSamples, arAwareTransform, c.imageConfig.PreEncodeImages) // Fetch the payloads and and deserialize them
	}()

	c.waitGroup = &wg
}

// Get a deserialized sample from the client
func (c *DatagoClient) GetSample() Sample {
	if c.cancel == nil && c.servedSamples == 0 {
		fmt.Println("Datago client not started. Starting it on the first sample, this adds some initial latency")
		fmt.Println("Please consider starting the client in anticipation by calling .Start()")
		c.Start()
	}

	if c.limit > 0 && c.servedSamples == c.limit {
		fmt.Println("Reached the limit of samples to serve, stopping the client")
		c.Stop()
		return Sample{}
	}

	if sample, ok := <-c.chanSamples; ok {
		c.servedSamples++
		return sample
	}

	fmt.Println("chanSamples closed, no more samples to serve")
	return Sample{}
}

// Stop the background downloads, will clear the memory and CPU footprint
func (c *DatagoClient) Stop() {
	fmt.Println("Stopping the datago client")

	// Signal the coroutines that next round should be a stop
	if c.cancel == nil {
		return // Already stopped
	}
	c.cancel()

	// Clear the channels, in case a commit is blocking
	go consumeChannel(c.chanPages)
	go consumeChannel(c.chanSampleMetadata)
	go consumeChannel(c.chanSamples)

	// Wait for all goroutines to finish
	if c.waitGroup != nil {
		c.waitGroup.Wait()
	}

	fmt.Println("Datago client stopped")
	c.cancel = nil
	c.context = nil
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Coroutines which will be running in the background

func (c *DatagoClient) asyncDispatch() {
	// Break down the page results and maintain a list of individual items to be processed

	for {
		select {
		case <-c.context.Done():
			close(c.chanSampleMetadata)
			return
		case page, open := <-c.chanPages:
			if !open {
				fmt.Println("No more metadata to fetch, wrapping up")
				close(c.chanSampleMetadata)
				return
			}

			for _, item := range page.samplesDataPointers {
				select {
				case <-c.context.Done():
					close(c.chanSampleMetadata)
					return
				case c.chanSampleMetadata <- item:
					// Item sent to the channel
				}
			}
		}
	}
}
