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

// TODO: Remove all the DB specific stuff and make it generic
type DatagoConfig struct {
	Sources             string
	SourceType          DatagoSourceType
	RequireImages       bool
	RequireEmbeddings   bool
	Tags                string
	TagsNE              string
	HasAttributes       string
	LacksAttributes     string
	HasMasks            string
	LacksMasks          string
	HasLatents          string
	LacksLatents        string
	CropAndResize       bool
	DefaultImageSize    int
	DownsamplingRatio   int
	MinAspectRatio      float64
	MaxAspectRatio      float64
	PreEncodeImages     bool
	Rank                uint32
	WorldSize           uint32
	PrefetchBufferSize  int
	SamplesBufferSize   int
	ConcurrentDownloads int
	PageSize            int
}

type DatagoClient struct {
	concurrency int

	context   context.Context
	waitGroup *sync.WaitGroup
	cancel    context.CancelFunc

	// Online transform parameters
	crop_and_resize    bool
	default_image_size int
	downsampling_ratio int
	min_aspect_ratio   float64
	max_aspect_ratio   float64
	pre_encode_images  bool

	// Flexible generator, backend and dispatch goroutines
	generator Generator
	backend   Backend

	// Channels	- these will be used to communicate between the background goroutines
	chanPages          chan Pages
	chanSampleMetadata chan SampleDataPointers
	chanSamples        chan Sample
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

func GetDefaultConfig() DatagoConfig {
	return DatagoConfig{
		Sources:             "",
		SourceType:          SourceTypeDB,
		RequireImages:       true,
		RequireEmbeddings:   false,
		Tags:                "",
		TagsNE:              "",
		HasAttributes:       "",
		LacksAttributes:     "",
		HasMasks:            "",
		LacksMasks:          "",
		HasLatents:          "",
		LacksLatents:        "",
		CropAndResize:       false,
		DefaultImageSize:    512,
		DownsamplingRatio:   16,
		MinAspectRatio:      0.5,
		MaxAspectRatio:      2.0,
		PreEncodeImages:     false,
		Rank:                0,
		WorldSize:           1,
		PrefetchBufferSize:  8,
		SamplesBufferSize:   8,
		ConcurrentDownloads: 2,
		PageSize:            20, // 1000 for a vectorDB, make this a default which depends on the source type
	}
}

// Create a new Dataroom Client
func GetClient(config DatagoConfig) *DatagoClient {

	// Create the generator and backend
	var generator Generator
	var backend Backend

	if config.SourceType == SourceTypeDB {
		fmt.Println("Creating a DB-backed dataloader")
		generator = newDatagoGeneratorDB(config)
		backend = BackendHTTP{config: &config}
	} else if config.SourceType == SourceTypeFileSystem {
		fmt.Println("Creating a FileSystem-backed dataloader")
		generator = newDatagoGeneratorFileSystem(config)
		backend = BackendFileSystem{config: &config}
	} else {
		// TODO: Handle other sources
		log.Panic("Unsupported source type at the moment")
	}

	// Create the client
	client := &DatagoClient{
		concurrency:        config.ConcurrentDownloads,
		chanPages:          make(chan Pages, 2),
		chanSampleMetadata: make(chan SampleDataPointers, config.PrefetchBufferSize),
		chanSamples:        make(chan Sample, config.SamplesBufferSize),

		crop_and_resize:    config.CropAndResize,
		default_image_size: config.DefaultImageSize,
		downsampling_ratio: config.DownsamplingRatio,
		min_aspect_ratio:   config.MinAspectRatio,
		max_aspect_ratio:   config.MaxAspectRatio,
		pre_encode_images:  config.PreEncodeImages,
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

	if c.crop_and_resize {
		fmt.Println("Cropping and resizing images")
		fmt.Println("Base image size | downsampling ratio | min | max:", c.default_image_size, c.downsampling_ratio, c.min_aspect_ratio, c.max_aspect_ratio)
		arAwareTransform = newARAwareTransform(c.default_image_size, c.downsampling_ratio, c.min_aspect_ratio, c.max_aspect_ratio)
	}

	if c.pre_encode_images {
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
		c.backend.collectSamples(c.chanSampleMetadata, c.chanSamples, arAwareTransform) // Fetch the payloads and and deserialize them
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
