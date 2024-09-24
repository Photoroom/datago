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
	Data           []uint8
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

// Create a new Dataroom Client
func GetClient(
	sources string,
	require_images, require_embeddings bool,
	tags, tags__ne string,
	has_attributes, lacks_attributes, has_masks, lacks_masks, has_latents, lacks_latents string,
	crop_and_resize bool, default_image_size, downsampling_ratio int,
	pre_encode_images bool,
	rank, world_size uint32,
	prefetch_buffer_size, samples_buffer_size, downloads_concurrency int) *DataroomClient {

	const PAGE_SIZE = "1000"

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
	fmt.Println("Rank | World size:", rank, world_size)

	fields := "attributes,image_direct_url"
	if has_latents != "" || has_masks != "" {
		fields += ",latents"
		fmt.Println("Including some latents:", has_latents, has_masks)
	}

	if tags != "" {
		fields += ",tags"
		fmt.Println("Including some tags:", tags)
	}

	if has_attributes != "" {
		fmt.Println("Including some attributes:", has_attributes)
	}

	if require_embeddings {
		fields += ",coca_embedding"
		fmt.Println("Including embeddings")
	}

	fmt.Println("Sources:", sources, "| Fields:", fields)

	// Define the query which will be the backbone of this DataroomClient instance
	request := PageRequest{
		fields:          fields,
		sources:         sanitizeStr(&sources),
		pageSize:        PAGE_SIZE,
		tags:            sanitizeStr(&tags),
		tagsNE:          sanitizeStr(&tags__ne),
		hasAttributes:   sanitizeStr(&has_attributes),
		lacksAttributes: sanitizeStr(&lacks_attributes),
		hasMasks:        sanitizeStr(&has_masks),
		lacksMasks:      sanitizeStr(&lacks_masks),
		hasLatents:      sanitizeStr(&has_latents),
		lacksLatents:    sanitizeStr(&lacks_latents),
	}

	client := &DataroomClient{
		concurrency:        downloads_concurrency,
		baseRequest:        *getHTTPRequest(api_url, api_key, request),
		chanPageResults:    make(chan Response, 2),
		chanSampleMetadata: make(chan SampleMetadata, prefetch_buffer_size),
		chanSamples:        make(chan Sample, samples_buffer_size),
		require_images:     require_images,
		require_embeddings: require_embeddings,
		has_masks:          strings.Split(has_masks, ","),
		has_latents:        strings.Split(has_latents, ","),
		crop_and_resize:    crop_and_resize,
		default_image_size: default_image_size,
		downsampling_ratio: downsampling_ratio,
		pre_encode_images:  pre_encode_images,
		min_aspect_ratio:   0.5,
		max_aspect_ratio:   2.0,
		sources:            sources,
		rank:               rank,
		world_size:         world_size,
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
