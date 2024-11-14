package datago

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// Interact with a DB to get payloads and process them
// Define a generator and a backend goroutine

// --- DB Communication structures ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type urlPayload struct {
	url     string
	content []byte
}

type urlLatent struct {
	URL        string `json:"file_direct_url"`
	LatentType string `json:"latent_type"`
	IsMask     bool   `json:"is_mask"`
}

type dbSampleMetadata struct {
	Id             string                 `json:"id"`
	Attributes     map[string]interface{} `json:"attributes"`
	DuplicateState int                    `json:"duplicate_state"`
	ImageDirectURL string                 `json:"image_direct_url"`
	Latents        []urlLatent            `json:"latents"`
	Tags           []string               `json:"tags"`
	CocaEmbedding  struct {
		Vector []float32 `json:"vector"`
	} `json:"coca_embedding"`
}

type dbResponse struct {
	Next             string             `json:"next"`
	DBSampleMetadata []dbSampleMetadata `json:"results"`
}

type dbRequest struct {
	fields   string
	sources  string
	pageSize string

	tags   string
	tagsNE string

	hasAttributes   string
	lacksAttributes string

	hasMasks   string
	lacksMasks string

	hasLatents    string
	lacksLatents  string
	returnLatents string

	minShortEdge string
	maxShortEdge string

	minPixelCount string
	maxPixelCount string

	randomSampling bool

	partitionsCount string
	partition       string
}

// -- Define the front end goroutine ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type SourceDBConfig struct {
	DataSourceConfig
	Sources           string `json:"sources"`
	RequireImages     bool   `json:"require_images"`
	RequireEmbeddings bool   `json:"require_embeddings"`
	Tags              string `json:"tags"`
	TagsNE            string `json:"tags_ne"`
	HasAttributes     string `json:"has_attributes"`
	LacksAttributes   string `json:"lacks_attributes"`
	HasMasks          string `json:"has_masks"`
	LacksMasks        string `json:"lacks_masks"`
	HasLatents        string `json:"has_latents"`
	LacksLatents      string `json:"lacks_latents"`

	ReturnLatents        string `json:"return_latents"`
	ReturnDuplicateState bool   `json:"return_duplicate_state"`

	MinShortEdge   int  `json:"min_short_edge"`
	MaxShortEdge   int  `json:"max_short_edge"`
	MinPixelCount  int  `json:"min_pixel_count"`
	MaxPixelCount  int  `json:"max_pixel_count"`
	RandomSampling bool `json:"random_sampling"`
}

func (c *SourceDBConfig) setDefaults() {
	c.PageSize = 512
	c.Rank = -1
	c.WorldSize = -1

	c.Sources = ""
	c.RequireImages = true
	c.RequireEmbeddings = false
	c.Tags = ""
	c.TagsNE = ""
	c.HasAttributes = ""
	c.LacksAttributes = ""
	c.HasMasks = ""
	c.LacksMasks = ""
	c.HasLatents = ""
	c.LacksLatents = ""
	c.ReturnLatents = ""
	c.ReturnDuplicateState = false

	c.MinShortEdge = -1
	c.MaxShortEdge = -1
	c.MinPixelCount = -1
	c.MaxPixelCount = -1
	c.RandomSampling = false

}

func (c *SourceDBConfig) getDbRequest() dbRequest {

	fields := "attributes,image_direct_url"
	if len(c.HasLatents) > 0 || len(c.HasMasks) > 0 {
		fields += ",latents"
		fmt.Println("Including some latents:", c.HasLatents, c.HasMasks)
	}

	if len(c.Tags) > 0 {
		fields += ",tags"
		fmt.Println("Including some tags:", c.Tags)
	}

	if len(c.HasLatents) > 0 {
		fmt.Println("Including some attributes:", c.HasLatents)
	}

	if c.RequireEmbeddings {
		fields += ",coca_embedding"
		fmt.Println("Including embeddings")
	}

	if c.ReturnDuplicateState {
		fields += ",duplicate_state"
		fmt.Println("Including duplicate state")
	}

	// Report some config data
	fmt.Println("Rank | World size:", c.Rank, c.WorldSize)
	fmt.Println("Sources:", c.Sources, "| Fields:", fields)

	sanitizeInt := func(val int) string {
		if val == -1 {
			return ""
		}
		return fmt.Sprintf("%d", val)
	}

	// Align rank and worldsize with the partitioning
	if c.WorldSize < 2 {
		// No partitioning
		c.WorldSize = -1
		c.Rank = -1
	}

	return dbRequest{
		fields:          fields,
		sources:         c.Sources,
		pageSize:        fmt.Sprintf("%d", c.PageSize),
		tags:            c.Tags,
		tagsNE:          c.TagsNE,
		hasAttributes:   c.HasAttributes,
		lacksAttributes: c.LacksAttributes,
		hasMasks:        c.HasMasks,
		lacksMasks:      c.LacksMasks,
		hasLatents:      c.HasLatents,
		lacksLatents:    c.LacksLatents,
		returnLatents:   c.HasLatents, // Could be exposed as it's done internally
		minShortEdge:    sanitizeInt(c.MinShortEdge),
		maxShortEdge:    sanitizeInt(c.MaxShortEdge),
		minPixelCount:   sanitizeInt(c.MinPixelCount),
		maxPixelCount:   sanitizeInt(c.MaxPixelCount),
		randomSampling:  c.RandomSampling,
		partitionsCount: sanitizeInt(c.WorldSize),
		partition:       sanitizeInt(c.Rank),
	}
}

func GetSourceDBConfig() SourceDBConfig {
	config := SourceDBConfig{}
	config.setDefaults()
	return config
}

type datagoGeneratorDB struct {
	baseRequest http.Request
	config      SourceDBConfig
}

func newDatagoGeneratorDB(config SourceDBConfig) datagoGeneratorDB {
	request := config.getDbRequest()

	api_key := os.Getenv("DATAROOM_API_KEY")
	if api_key == "" {
		log.Panic("DATAROOM_API_KEY is not set")
	}

	api_url := os.Getenv("DATAROOM_API_URL")
	if api_url == "" {
		log.Panic("DATAROOM_API_URL is not set")
	}

	fmt.Println("Dataroom API URL:", api_url)
	fmt.Println("Dataroom API KEY last characters:", getLast5Chars(api_key))

	return datagoGeneratorDB{baseRequest: *getHTTPRequest(api_url, api_key, request), config: config}
}

func (f datagoGeneratorDB) generatePages(ctx context.Context, chanPages chan Pages) {
	// Fetch pages from the API, and feed the results to the items channel
	// This is meant to be run in a goroutine
	http_client := http.Client{Timeout: 30 * time.Second}
	max_retries := 10

	fetch_new_page := func() (*dbResponse, error) {
		resp, err := http_client.Do(&f.baseRequest)

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
			return nil, fmt.Errorf("error reading dbResponse body - renewing HTTP client. %s", err)
		}

		// Unmarshal JSON dbResponse
		var data dbResponse
		if err = json.Unmarshal(body, &data); err != nil {
			return nil, err
		}

		return &data, nil
	}

	for {
		select {
		case <-ctx.Done():
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
				if len(data.DBSampleMetadata) > 0 {
					// TODO: There's probably a better way to do this
					samplesDataPointers := make([]SampleDataPointers, len(data.DBSampleMetadata))

					for i, sample := range data.DBSampleMetadata {
						samplesDataPointers[i] = sample
					}

					chanPages <- Pages{samplesDataPointers}
				}

				// Check if there are more pages to fetch
				if data.Next == "" {
					fmt.Println("No more pages to fetch, wrapping up")
					close(chanPages)
					return
				}

				// Else fetch the next page
				authentication := f.baseRequest.Header.Get("Authorization")
				nextURL, _ := http.NewRequest("GET", data.Next, nil)
				nextURL.Header.Add("Authorization", authentication)
				f.baseRequest = *nextURL

				// Break the loop on success, gives us the opportunity to check whether the context has closed
				valid_page = true
				break
			}

			// Check if we consumed all the retries
			if !valid_page {
				fmt.Println("Too many errors fetching new pages, wrapping up")
				close(chanPages)
				return
			}
		}
	}
}
