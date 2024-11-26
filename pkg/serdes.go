package datago

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
)

func readBodyBuffered(resp *http.Response) ([]byte, error) {
	// Use a bytes.Buffer to accumulate the response body
	// Faster than the default ioutil.ReadAll which reallocates
	var body bytes.Buffer

	bufferSize := 2048 * 1024 // 2MB

	// Create a fixed-size buffer for reading
	local_buffer := make([]byte, bufferSize)

	for {
		n, err := resp.Body.Read(local_buffer)
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n > 0 {
			body.Write(local_buffer[:n])
		}
		if err == io.EOF {
			break
		}
	}
	return body.Bytes(), nil
}

func imageFromBuffer(buffer []byte, transform *ARAwareTransform, aspect_ratio float64, pre_encode_image bool, is_mask bool) (*ImagePayload, float64, error) {
	// Decode the image payload using vips
	img, err := vips.NewImageFromBuffer(buffer)
	if err != nil {
		return nil, -1., err
	}

	err = img.AutoRotate()
	if err != nil {
		return nil, -1., err
	}

	// Optionally crop and resize the image on the fly. Save the aspect ratio in the process for future use
	original_width, original_height := img.Width(), img.Height()

	if transform != nil {
		aspect_ratio, err = transform.cropAndResizeToClosestAspectRatio(img, aspect_ratio)
		if err != nil {
			return nil, -1., err
		}
	}

	width, height := img.Width(), img.Height()

	// If the image is 4 channels, we need to drop the alpha channel
	if img.Bands() == 4 {
		err = img.Flatten(&vips.Color{R: 255, G: 255, B: 255}) // Flatten with white background
		if err != nil {
			fmt.Println("Error flattening image:", err)
			return nil, -1., err
		}
		fmt.Println("Image flattened")
	}

	// If the image is not a mask but is 1 channel, we want to convert it to 3 channels
	if (img.Bands() == 1) && !is_mask {
		err = img.ToColorSpace(vips.InterpretationSRGB)
		if err != nil {
			fmt.Println("Error converting to sRGB:", err)
			return nil, -1., err
		}
	}

	// If requested, re-encode the image to a jpg or png
	var img_bytes []byte
	var channels int
	if pre_encode_image {
		if err != nil {
			return nil, -1., err
		}

		if img.Bands() == 3 {
			// Re-encode the image to a jpg
			img_bytes, _, err = img.ExportJpeg(&vips.JpegExportParams{Quality: 95})
			if err != nil {
				return nil, -1., err
			}
		} else {
			// Re-encode the image to a png
			img_bytes, _, err = img.ExportPng(vips.NewPngExportParams())
			if err != nil {
				return nil, -1., err
			}
		}
		channels = -1 // Signal that we have encoded the image
	} else {
		img_bytes, err = img.ToBytes()
		if err != nil {
			return nil, -1., err
		}
		channels = img.Bands()
	}

	img_payload := ImagePayload{
		Data:           img_bytes,
		OriginalHeight: original_height,
		OriginalWidth:  original_width,
		Height:         height,
		Width:          width,
		Channels:       channels,
	}

	return &img_payload, aspect_ratio, nil
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
		Source:           sample_result.Source,
		Attributes:       sample_result.Attributes,
		Image:            *img_payload,
		Latents:          latents,
		Masks:            masks,
		AdditionalImages: additional_images,
		Tags:             sample_result.Tags,
		CocaEmbedding:    cocaEmbedding}
}
