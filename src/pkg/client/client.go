package datago

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
)

// --- Sample data structures - these will be exposed to the Python world ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type LatentPayload struct {
	Data    []byte
	Len     int
	DataPtr uintptr
}

type ImagePayload struct {
	Data           []byte
	OriginalHeight int // Good indicator of the image frequency dbResponse at the current resolution
	OriginalWidth  int
	Height         int // Useful to decode the current payload
	Width          int
	Channels       int
	DataPtr        uintptr
}

type Sample struct {
	ID               string
	Source           string
	Attributes       map[string]interface{}
	Image            ImagePayload
	Masks            map[string]ImagePayload
	AdditionalImages map[string]ImagePayload
	Latents          map[string]LatentPayload
	CocaEmbedding    []float32
	Tags             []string
}

type URLPayload struct {
	url     string
	content []byte
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Public interface for this package, will be reflected in the python bindings

type DatagoSourceType string

const (
	SourceTypeDB           DatagoSourceType = "DB"
	SourceTypeLocalStorage DatagoSourceType = "LocalStorage"
	// incoming: object storage
)

type DataroomClientConfig struct {
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

type DataroomClient struct {
	concurrency int
	baseRequest http.Request

	context   context.Context
	waitGroup *sync.WaitGroup
	cancel    context.CancelFunc

	// Request parameters
	sources            string
	require_images     bool
	require_embeddings bool
	has_masks          []string
	has_latents        []string
	rank               uint32
	world_size         uint32

	// Online transform parameters
	crop_and_resize    bool
	default_image_size int
	downsampling_ratio int
	min_aspect_ratio   float64
	max_aspect_ratio   float64
	pre_encode_images  bool

	// Flexible frontend, backend and dispatch goroutines
	frontend Frontend

	// Channels	- these will be used to communicate between the background goroutines
	chanPageResults      chan dbResponse
	chandbSampleMetadata chan dbSampleMetadata
	chanSamples          chan Sample
}

// -----------------------------------------------------------------------------------------------------------------
// Define the interfaces that the different features will needd to follow

// The asyncFrontend will be responsible for producing pages of metadata which can be dispatched
// to the asyncDispatch goroutine. The metadata will be used to fetch the actual payloads

type Frontend interface {
	collectPages(ctx context.Context, chanPageResults chan dbResponse)
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

func GetDefaultConfig() DataroomClientConfig {
	return DataroomClientConfig{
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
		PageSize:            1000,
	}
}

// Create a new Dataroom Client
func GetClient(config DataroomClientConfig) *DataroomClient {

	// Create the frontend
	var frontend Frontend
	if config.SourceType == SourceTypeDB {
		frontend = newDatagoFrontendDB(config)
	} else {
		// TODO: Handle other sources
		log.Panic("Unsupported source type at the moment")
	}

	// Create the client
	client := &DataroomClient{
		concurrency:          config.ConcurrentDownloads,
		chanPageResults:      make(chan dbResponse, 2),
		chandbSampleMetadata: make(chan dbSampleMetadata, config.PrefetchBufferSize),
		chanSamples:          make(chan Sample, config.SamplesBufferSize),
		require_images:       config.RequireImages,
		require_embeddings:   config.RequireEmbeddings,
		has_masks:            strings.Split(config.HasMasks, ","),
		has_latents:          strings.Split(config.HasLatents, ","),
		crop_and_resize:      config.CropAndResize,
		default_image_size:   config.DefaultImageSize,
		downsampling_ratio:   config.DownsamplingRatio,
		min_aspect_ratio:     config.MinAspectRatio,
		max_aspect_ratio:     config.MaxAspectRatio,
		pre_encode_images:    config.PreEncodeImages,
		sources:              config.Sources,
		rank:                 config.Rank,
		world_size:           config.WorldSize,
		context:              nil,
		cancel:               nil,
		waitGroup:            nil,
		frontend:             frontend,
	}

	// Make sure that the client will be Stopped() upon destruction
	runtime.SetFinalizer(client, func(r *DataroomClient) {
		r.Stop()
	})

	return client
}

// Start the background downloads, make it ready to serve samples. Will grow the memory and CPU footprint
func (c *DataroomClient) Start() {
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
		c.frontend.collectPages(c.context, c.chanPageResults) // Collect the root data source pages
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.asyncDispatch() // Dispatch the content of the pages to the items channel
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.asyncBackend(arAwareTransform) // Fetch the payloads and and deserialize them
	}()

	c.waitGroup = &wg
}

// Get a deserialized sample from the client
func (c *DataroomClient) GetSample() Sample {
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
func (c *DataroomClient) Stop() {
	fmt.Println("Stopping the dataroom client")

	// Signal the coroutines that next round should be a stop
	if c.cancel == nil {
		return // Already stopped
	}
	c.cancel()

	// Clear the channels, in case a commit is blocking
	go consumeChannel(c.chanPageResults)
	go consumeChannel(c.chandbSampleMetadata)
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

func (c *DataroomClient) asyncFrontend() {

}

func (c *DataroomClient) asyncDispatch() {
	// Break down the page results and maintain a list of individual items to be processed

	for {
		select {
		case <-c.context.Done():
			fmt.Println("Metadata fetch goroutine wrapping up")
			close(c.chandbSampleMetadata)
			return
		case page, open := <-c.chanPageResults:
			if !open {
				fmt.Println("No more metadata to fetch, wrapping up")
				close(c.chandbSampleMetadata)
				return
			}

			for _, item := range page.DBSampleMetadata {
				// Skip the sample if multi-rank is enabled and the rank is not the one we're interested in
				// NOTE: if the front end is a DB, this is a wasteful way to distribute the work
				//       since we waste most of the page we fetched
				if c.world_size > 1 && computeFNVHash32(item.Id)%c.world_size != c.rank {
					continue
				}

				select {
				case <-c.context.Done():
					fmt.Println("Metadata fetch goroutine wrapping up")
					close(c.chandbSampleMetadata)
					return
				case c.chandbSampleMetadata <- item:
					// Item sent to the channel
				}
			}
		}
	}
}

func (c *DataroomClient) asyncBackend(transform *ARAwareTransform) {
	ack_channel := make(chan bool)

	sampleWorker := func(client *DataroomClient, transform *ARAwareTransform) {
		// One HHTP client per goroutine, make sure we don't run into racing conditions when renewing
		http_client := http.Client{Timeout: 30 * time.Second}

		for {
			item_to_fetch, open := <-client.chandbSampleMetadata
			if !open {
				ack_channel <- true
				return
			}

			sample := fetchSample(client, &http_client, item_to_fetch, transform)
			if sample != nil {
				client.chanSamples <- *sample
			}
		}
	}

	// Start the workers and work on the metadata channel
	for i := 0; i < c.concurrency; i++ {
		go sampleWorker(c, transform)
	}

	// Wait for all the workers to be done or overall context to be cancelled
	for i := 0; i < c.concurrency; i++ {
		<-ack_channel
	}
	close(c.chanSamples)
	fmt.Println("No more items to serve, wrapping up")
}
