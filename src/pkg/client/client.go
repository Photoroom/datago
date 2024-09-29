package datago

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// --- DB Communication structures ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type URLLatent struct {
	URL        string `json:"file_direct_url"`
	LatentType string `json:"latent_type"`
	IsMask     bool   `json:"is_mask"`
}

type SampleMetadata struct {
	Id             string                 `json:"id"`
	Attributes     map[string]interface{} `json:"attributes"`
	ImageDirectURL string                 `json:"image_direct_url"`
	Latents        []URLLatent            `json:"latents"`
	Tags           []string               `json:"tags"`
	CocaEmbedding  struct {
		Vector []float32 `json:"vector"`
	} `json:"coca_embedding"`
}

type Response struct {
	Next           string           `json:"next"`
	SampleMetadata []SampleMetadata `json:"results"`
}

type PageRequest struct {
	fields   string
	sources  string
	pageSize string

	tags   string
	tagsNE string

	hasAttributes   string
	lacksAttributes string

	hasMasks   string
	lacksMasks string

	hasLatents   string
	lacksLatents string
}

// --- Sample data structures - these will be exposed to the Python world ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type LatentPayload struct {
	Data    []byte
	Len     int
	DataPtr uintptr
}

type ImagePayload struct {
	Data           []byte
	OriginalHeight int // Good indicator of the image frequency response at the current resolution
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

	// Channels	- these will be used to communicate between the background goroutines
	chanPageResults    chan Response
	chanSampleMetadata chan SampleMetadata
	chanSamples        chan Sample
}

type DataroomClientConfig struct {
	Sources             string
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

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

func GetDefaultConfig() DataroomClientConfig {
	return DataroomClientConfig{
		Sources:             "",
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

func (c *DataroomClientConfig) getPageRequest() PageRequest {

	fields := "attributes,image_direct_url"
	if c.HasLatents != "" || c.HasMasks != "" {
		fields += ",latents"
		fmt.Println("Including some latents:", c.HasLatents, c.HasMasks)
	}

	if c.Tags != "" {
		fields += ",tags"
		fmt.Println("Including some tags:", c.Tags)
	}

	if c.HasLatents != "" {
		fmt.Println("Including some attributes:", c.HasLatents)
	}

	if c.RequireEmbeddings {
		fields += ",coca_embedding"
		fmt.Println("Including embeddings")
	}

	// Report some config data
	fmt.Println("Rank | World size:", c.Rank, c.WorldSize)
	fmt.Println("Sources:", c.Sources, "| Fields:", fields)

	return PageRequest{
		fields:          fields,
		sources:         sanitizeStr(&c.Sources),
		pageSize:        fmt.Sprintf("%d", c.PageSize),
		tags:            sanitizeStr(&c.Tags),
		tagsNE:          sanitizeStr(&c.TagsNE),
		hasAttributes:   sanitizeStr(&c.HasAttributes),
		lacksAttributes: sanitizeStr(&c.LacksAttributes),
		hasMasks:        sanitizeStr(&c.HasMasks),
		lacksMasks:      sanitizeStr(&c.LacksMasks),
		hasLatents:      sanitizeStr(&c.HasLatents),
		lacksLatents:    sanitizeStr(&c.LacksLatents),
	}
}

// Create a new Dataroom Client
func GetClient(config DataroomClientConfig) *DataroomClient {

	api_key := os.Getenv("DATAROOM_API_KEY")
	if api_key == "" {
		log.Panic("DATAROOM_API_KEY is not set")
	}

	api_url := os.Getenv("DATAROOM_API_URL")
	if api_url == "" {
		api_url = "https://dataroomv2.photoroom.com/api/"
	}

	fmt.Println("Dataroom API URL:", api_url)
	fmt.Println("Dataroom API KEY last characters:", getLast5Chars(api_key))

	// Define the query which will be the backbone of this DataroomClient instance
	request := config.getPageRequest()

	client := &DataroomClient{
		concurrency:        config.ConcurrentDownloads,
		baseRequest:        *getHTTPRequest(api_url, api_key, request),
		chanPageResults:    make(chan Response, 2),
		chanSampleMetadata: make(chan SampleMetadata, config.PrefetchBufferSize),
		chanSamples:        make(chan Sample, config.SamplesBufferSize),
		require_images:     config.RequireImages,
		require_embeddings: config.RequireEmbeddings,
		has_masks:          strings.Split(config.HasMasks, ","),
		has_latents:        strings.Split(config.HasLatents, ","),
		crop_and_resize:    config.CropAndResize,
		default_image_size: config.DefaultImageSize,
		downsampling_ratio: config.DownsamplingRatio,
		min_aspect_ratio:   config.MinAspectRatio,
		max_aspect_ratio:   config.MaxAspectRatio,
		pre_encode_images:  config.PreEncodeImages,
		sources:            config.Sources,
		rank:               config.Rank,
		world_size:         config.WorldSize,
		context:            nil,
		cancel:             nil,
		waitGroup:          nil,
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
		c.collectPages() // Fetch pages from the DB in the background
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.collectMetadata() // Dispatch the content of the pages to the items channel
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.collectItems(arAwareTransform) // Fetch the payloads and and deserialize them
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

func (c *DataroomClient) collectPages() {
	// Fetch pages from the API, and feed the results to the items channel
	// This is meant to be run in a goroutine
	http_client := http.Client{Timeout: 30 * time.Second}
	max_retries := 10

	fetch_new_page := func() (*Response, error) {
		resp, err := http_client.Do(&c.baseRequest)

		if err != nil {
			return nil, err
		}

		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("error fetching page: %s", resp.Status)
		}

		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			// Renew the HTTP client, server closed the connection
			http_client = http.Client{Timeout: 30 * time.Second}
			return nil, fmt.Errorf("error reading response body - renewing HTTP client. %s", err)
		}

		// Unmarshal JSON response
		var data Response
		if err = json.Unmarshal(body, &data); err != nil {
			return nil, err
		}
		return &data, nil
	}

	for {
		select {
		case <-c.context.Done():
			fmt.Println("Pages fetch goroutine wrapping up")
			return

		default:
			valid_page := false

			for i := 0; i < max_retries; i++ {
				// Try to fetch a new page, could go wrong in many ways
				data, err := fetch_new_page()

				if err != nil {
					// Retry loop
					log.Print("Error fetching page: ", err)
					continue
				}

				// Commit the possible results to the downstream goroutines
				if len(data.SampleMetadata) > 0 {
					c.chanPageResults <- *data
				}

				// Check if there are more pages to fetch
				if data.Next == "" {
					fmt.Println("No more pages to fetch, wrapping up")
					close(c.chanPageResults)
					return
				}

				// Else fetch the next page
				authentication := c.baseRequest.Header.Get("Authorization")
				nextURL, _ := http.NewRequest("GET", data.Next, nil)
				nextURL.Header.Add("Authorization", authentication)
				c.baseRequest = *nextURL

				// Break the loop on success, gives us the opportunity to check whether the context has closed
				valid_page = true
				break
			}

			// Check if we consumed all the retries
			if !valid_page {
				fmt.Println("Too many errors fetching new pages, wrapping up")
				close(c.chanPageResults)
				return
			}
		}
	}
}

func (c *DataroomClient) collectMetadata() {
	// Break down the page results and maintain a list of individual items to be downloaded
	// This is meant to be run in a goroutine

	for {
		select {
		case <-c.context.Done():
			fmt.Println("Metadata fetch goroutine wrapping up")
			close(c.chanSampleMetadata)
			return
		case page, open := <-c.chanPageResults:
			if !open {
				fmt.Println("No more metadata to fetch, wrapping up")
				close(c.chanSampleMetadata)
				return
			}

			for _, item := range page.SampleMetadata {
				// Skip the sample if multi-rank is enabled and the rank is not the one we're interested in
				if c.world_size > 1 && computeFNVHash32(item.Id)%c.world_size != c.rank {
					continue
				}

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

func (c *DataroomClient) collectItems(transform *ARAwareTransform) {
	ack_channel := make(chan bool)

	sampleWorker := func(client *DataroomClient, transform *ARAwareTransform) {
		// One HHTP client per goroutine, make sure we don't run into racing conditions when renewing
		http_client := http.Client{Timeout: 30 * time.Second}

		for {
			item_to_fetch, open := <-client.chanSampleMetadata
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
