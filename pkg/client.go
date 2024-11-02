package datago

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"sync"

	"github.com/davidbyttow/govips/v2/vips"
)

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Public interface for this package, will be reflected in the python bindings

type DatagoSourceType string

const (
	SourceTypeDB         DatagoSourceType = "DB"
	SourceTypeFileSystem DatagoSourceType = "FileSystem"
	// incoming: object storage
)

type DataSourceConfig interface{} // FIXME: This is not compatible with python bindings

type ImageTransformConfig struct {
	CropAndResize     bool
	DefaultImageSize  int
	DownsamplingRatio int
	MinAspectRatio    float64
	MaxAspectRatio    float64
	PreEncodeImages   bool
}

func (c *ImageTransformConfig) SetDefaults() {
	c.DefaultImageSize = 512
	c.DownsamplingRatio = 16
	c.MinAspectRatio = 0.5
	c.MaxAspectRatio = 2.0
	c.PreEncodeImages = false
}

type DatagoConfig struct {
	SourceType         DatagoSourceType `default:"DB"`
	SourceConfig       DataSourceConfig
	ImageConfig        ImageTransformConfig
	PrefetchBufferSize int
	SamplesBufferSize  int
	Concurrency        int
}

func (c *DatagoConfig) SetDefaults() {
	c.SourceType = SourceTypeDB

	dbConfig := GeneratorDBConfig{}
	dbConfig.SetDefaults()
	c.SourceConfig = dbConfig

	c.ImageConfig.SetDefaults()
	c.PrefetchBufferSize = 64
	c.SamplesBufferSize = 32
	c.Concurrency = 64
}

type DatagoClient struct {
	context   context.Context
	waitGroup *sync.WaitGroup
	cancel    context.CancelFunc

	ImageConfig ImageTransformConfig

	// Flexible generator, backend and dispatch goroutines
	generator Generator
	backend   Backend

	// Channels	- these will be used to communicate between the background goroutines
	chanPages          chan Pages
	chanSampleMetadata chan SampleDataPointers
	chanSamples        chan Sample
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Create a new Dataroom Client
func GetClient(config DatagoConfig) *DatagoClient {
	// Create the generator and backend
	var generator Generator
	var backend Backend

	if config.SourceType == SourceTypeDB {
		fmt.Println("Creating a DB-backed dataloader")
		db_config := config.SourceConfig.(GeneratorDBConfig)
		generator = newDatagoGeneratorDB(db_config)
		backend = BackendHTTP{config: &db_config, concurrency: config.Concurrency}
	} else if config.SourceType == SourceTypeFileSystem {
		fmt.Println("Creating a FileSystem-backed dataloader")
		fs_config := config.SourceConfig.(GeneratorFileSystemConfig)
		generator = newDatagoGeneratorFileSystem(fs_config)
		backend = BackendFileSystem{config: &config, concurrency: config.Concurrency}
	} else {
		// TODO: Handle other sources
		log.Panic("Unsupported source type at the moment")
	}

	// Create the client
	client := &DatagoClient{
		chanPages:          make(chan Pages, 2),
		chanSampleMetadata: make(chan SampleDataPointers, config.PrefetchBufferSize),
		chanSamples:        make(chan Sample, config.SamplesBufferSize),
		ImageConfig:        config.ImageConfig,
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

// Start the background downloads, make it ready to serve samples. Will grow the memory and CPU footprint
func (c *DatagoClient) Start() {
	if c.context == nil || c.cancel == nil {
		// Get a context and a cancel function to stop the background goroutines and gracefully handle
		// interruptions at during http round trips
		c.context, c.cancel = context.WithCancel(context.Background())
	}

	debug.SetGCPercent(10) // Invoke GC 10x more often

	vips.LoggingSettings(func(domain string, level vips.LogLevel, msg string) {
		fmt.Println(domain, level, msg)
	}, vips.LogLevelWarning)
	vips.Startup(nil) // Initialize the vips library, image processing backend

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

	if c.ImageConfig.CropAndResize {
		fmt.Println("Cropping and resizing images")
		fmt.Println("Base image size | downsampling ratio | min | max:", c.ImageConfig.DefaultImageSize, c.ImageConfig.DownsamplingRatio, c.ImageConfig.MinAspectRatio, c.ImageConfig.MaxAspectRatio)
		arAwareTransform = newARAwareTransform(c.ImageConfig)
	}

	if c.ImageConfig.PreEncodeImages {
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
		c.backend.collectSamples(c.chanSampleMetadata, c.chanSamples, arAwareTransform, c.ImageConfig.PreEncodeImages) // Fetch the payloads and and deserialize them
	}()

	c.waitGroup = &wg
}

// Get a deserialized sample from the client
func (c *DatagoClient) GetSample() Sample {
	if c.cancel == nil {
		fmt.Println("Dataroom client not started. Starting it on the first sample, this adds some initial latency")
		fmt.Println("Please consider starting the client in anticipation by calling .Start()")
		c.Start()
	}

	if sample, ok := <-c.chanSamples; ok {
		return sample
	}
	fmt.Println("chanSamples closed, no more samples to serve")
	return Sample{}
}

// Stop the background downloads, will clear the memory and CPU footprint
func (c *DatagoClient) Stop() {
	fmt.Println("Stopping the dataroom client")

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

	fmt.Println("Dataroom client stopped")
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
			fmt.Println("Metadata fetch goroutine wrapping up")
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
					fmt.Println("Metadata fetch goroutine wrapping up")
					close(c.chanSampleMetadata)
					return
				case c.chanSampleMetadata <- item:
					// Item sent to the channel
				}
			}
		}
	}
}
