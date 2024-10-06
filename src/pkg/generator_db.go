package datago

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
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

	hasLatents   string
	lacksLatents string
}

// -- Define the front end goroutine ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type GeneratorDBConfig struct {
	// Request parameters
	Sources             string
	RequireImages       bool
	RequireEmbeddings   bool
	Tags                []string
	TagsNE              []string
	HasAttributes       []string
	LacksAttributes     []string
	HasMasks            []string
	LacksMasks          []string
	HasLatents          []string
	LacksLatents        []string
	ConcurrentDownloads int
	PageSize            int
	Rank                uint32
	WorldSize           uint32
}

func (c *GeneratorDBConfig) getDbRequest() dbRequest {

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

	// Report some config data
	fmt.Println("Rank | World size:", c.Rank, c.WorldSize)
	fmt.Println("Sources:", c.Sources, "| Fields:", fields)

	return dbRequest{
		fields:          fields,
		sources:         c.Sources,
		pageSize:        fmt.Sprintf("%d", c.PageSize),
		tags:            strings.Join(c.Tags, ","),
		tagsNE:          strings.Join(c.TagsNE, ","),
		hasAttributes:   strings.Join(c.HasAttributes, ","),
		lacksAttributes: strings.Join(c.LacksAttributes, ","),
		hasMasks:        strings.Join(c.HasMasks, ","),
		lacksMasks:      strings.Join(c.LacksMasks, ","),
		hasLatents:      strings.Join(c.HasLatents, ","),
		lacksLatents:    strings.Join(c.LacksLatents, ","),
	}
}

func GetDefaultDBConfig() GeneratorDBConfig {
	return GeneratorDBConfig{
		Sources:             "",
		RequireImages:       true,
		RequireEmbeddings:   false,
		Tags:                []string{},
		TagsNE:              []string{},
		HasAttributes:       []string{},
		LacksAttributes:     []string{},
		HasMasks:            []string{},
		LacksMasks:          []string{},
		HasLatents:          []string{},
		LacksLatents:        []string{},
		Rank:                0,
		WorldSize:           0,
		ConcurrentDownloads: 1,
		PageSize:            1000,
	}
}

type datagoGeneratorDB struct {
	baseRequest http.Request
	config      GeneratorDBConfig
}

func newDatagoGeneratorDB(config GeneratorDBConfig) datagoGeneratorDB {
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

	generatorDBConfig := GeneratorDBConfig{
		RequireImages:     config.RequireImages,
		RequireEmbeddings: config.RequireEmbeddings,
		HasMasks:          config.HasMasks,
		LacksMasks:        config.LacksMasks,
		HasLatents:        config.HasLatents,
		LacksLatents:      config.LacksLatents,
		Sources:           config.Sources,
		Rank:              config.Rank,
		WorldSize:         config.WorldSize,
	}

	return datagoGeneratorDB{baseRequest: *getHTTPRequest(api_url, api_key, request), config: generatorDBConfig}
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
