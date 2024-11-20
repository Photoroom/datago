package datago

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

type BackendHTTP struct {
	config      *SourceDBConfig
	concurrency int
}

func (b BackendHTTP) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, pre_encode_images bool) {

	ack_channel := make(chan bool)

	sampleWorker := func() {
		// One HHTP client per goroutine, make sure we don't run into racing conditions when renewing
		http_client := http.Client{Timeout: 30 * time.Second}

		for {
			item_to_fetch, open := <-chanSampleMetadata
			if !open {
				ack_channel <- true
				return
			}

			// Cast the item to fetch to the correct type
			http_sample, ok := item_to_fetch.(dbSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample := fetchSample(b.config, &http_client, http_sample, transform, pre_encode_images)
			if sample != nil {
				chanSamples <- *sample
			}
		}
	}

	// Start the workers and work on the metadata channel
	for i := 0; i < b.concurrency; i++ {
		go sampleWorker()
	}

	// Wait for all the workers to be done or overall context to be cancelled
	for i := 0; i < b.concurrency; i++ {
		<-ack_channel
	}
	close(chanSamples)
}

func fetchURL(client *http.Client, url string, retries int) (urlPayload, error) {
	// Helper to fetch a binary payload from a URL
	err_msg := ""

	for i := 0; i < retries; i++ {
		resp, err := client.Get(url)
		if err != nil {
			if i == retries-1 {
				err_msg = fmt.Sprintf("failed to fetch %s %s", url, err)
			}
			exponentialBackoffWait(i)
			continue
		}
		defer resp.Body.Close()

		body_bytes, err := readBodyBuffered(resp)
		if err != nil {
			// Renew the http client, not a shared resource
			client = &http.Client{Timeout: 30 * time.Second}
			exponentialBackoffWait(i)
			continue
		}

		return urlPayload{url: url, content: body_bytes}, nil
	}

	return urlPayload{url: url, content: nil}, fmt.Errorf("%s", err_msg)
}

func fetchImage(client *http.Client, url string, retries int, transform *ARAwareTransform, aspect_ratio float64, pre_encode_image bool, is_mask bool) (*ImagePayload, float64, error) {
	err_report := fmt.Errorf("failed fetching image %s", url)

	for i := 0; i < retries; i++ {
		// Get the raw image payload
		resp, err := client.Get(url)
		if err != nil {
			err_report = err
			exponentialBackoffWait(i)

			// Renew the client in case the connection was closed
			client = &http.Client{Timeout: 30 * time.Second}
			continue
		}
		defer resp.Body.Close()

		body_bytes, err := readBodyBuffered(resp)
		if err != nil {
			err_report = err
			exponentialBackoffWait(i)
			continue
		}

		// Decode into a flat buffer using vips
		img_payload_ptr, aspect_ratio, err := imageFromBuffer(body_bytes, transform, aspect_ratio, pre_encode_image, is_mask)
		if err != nil {
			break
		}
		return img_payload_ptr, aspect_ratio, nil
	}
	return nil, -1., err_report
}

func fetchSample(config *SourceDBConfig, http_client *http.Client, sample_result dbSampleMetadata, transform *ARAwareTransform, pre_encode_image bool) *Sample {
	// Per sample work:
	// - fetch the raw payloads
	// - deserialize / decode, depending on the types
	// return the result to the samples channel

	retries := 5
	img_payload := &ImagePayload{}

	aspect_ratio := -1. // Not initialized to begin with

	// Base image
	if config.RequireImages {
		base_image, new_aspect_ratio, err := fetchImage(http_client, sample_result.ImageDirectURL, retries, transform, aspect_ratio, pre_encode_image, false)

		if err != nil {
			fmt.Println("Error fetching image:", sample_result.Id)
			return nil
		} else {
			img_payload = base_image
			aspect_ratio = new_aspect_ratio
		}
	}

	// Latents
	latents := make(map[string]LatentPayload)
	masks := make(map[string]ImagePayload)
	additional_images := make(map[string]ImagePayload)

	for _, latent := range sample_result.Latents {
		if strings.Contains(latent.LatentType, "image") && !strings.Contains(latent.LatentType, "latent_") {
			// Image types, registered as latents but they need to be jpg-decoded
			new_image, _, err := fetchImage(http_client, latent.URL, retries, transform, aspect_ratio, pre_encode_image, false)
			if err != nil {
				fmt.Println("Error fetching masked image:", sample_result.Id, latent.LatentType)
				return nil
			}

			additional_images[latent.LatentType] = *new_image
		} else if latent.IsMask {
			// Mask types, registered as latents but they need to be png-decoded
			mask_ptr, _, err := fetchImage(http_client, latent.URL, retries, transform, aspect_ratio, pre_encode_image, true)
			if err != nil {
				fmt.Println("Error fetching mask:", sample_result.Id, latent.LatentType)
				return nil
			}
			masks[latent.LatentType] = *mask_ptr
		} else {
			// Vanilla latents, pure binary payloads
			latent_payload, err := fetchURL(http_client, latent.URL, retries)
			if err != nil {
				fmt.Println("Error fetching latent:", err)
				return nil
			}

			latents[latent.LatentType] = LatentPayload{
				latent_payload.content,
				len(latent_payload.content),
			}
		}
	}

	// Optional embeddings
	var cocaEmbedding []float32
	if config.RequireEmbeddings {
		cocaEmbedding = sample_result.CocaEmbedding.Vector
	}

	return &Sample{ID: sample_result.Id,
		Source:           config.Sources,
		Attributes:       sample_result.Attributes,
		Image:            *img_payload,
		Latents:          latents,
		Masks:            masks,
		AdditionalImages: additional_images,
		Tags:             sample_result.Tags,
		CocaEmbedding:    cocaEmbedding}
}
