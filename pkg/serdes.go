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
	localBuffer := make([]byte, bufferSize)

	for {
		n, err := resp.Body.Read(localBuffer)
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n > 0 {
			body.Write(localBuffer[:n])
		}
		if err == io.EOF {
			break
		}
	}
	return body.Bytes(), nil
}

func imageFromBuffer(buffer []byte, transform *ARAwareTransform, aspectRatio float64, encodeImage bool, isMask bool) (*ImagePayload, float64, error) {
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
	originalWidth, originalHeight := img.Width(), img.Height()

	if transform != nil {
		aspectRatio, err = transform.cropAndResizeToClosestAspectRatio(img, aspectRatio)
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
	if (img.Bands() == 1) && !isMask {
		err = img.ToColorSpace(vips.InterpretationSRGB)
		if err != nil {
			fmt.Println("Error converting to sRGB:", err)
			return nil, -1., err
		}
	}

	// If requested, re-encode the image to a jpg or png
	var imgBytes []byte
	var channels int
	var bitDepth int

	if encodeImage {
		if err != nil {
			return nil, -1., err
		}

		if img.Bands() == 3 {
			// Re-encode the image to a jpg
			imgBytes, _, err = img.ExportJpeg(&vips.JpegExportParams{Quality: 95})
			if err != nil {
				return nil, -1., err
			}
		} else {
			// Re-encode the image to a png
			imgBytes, _, err = img.ExportPng(vips.NewPngExportParams())
			if err != nil {
				return nil, -1., err
			}
		}
		channels = -1 // Signal that we have encoded the image
	} else {
		imgBytes, err = img.ToBytes()
		if err != nil {
			return nil, -1., err
		}
		channels = img.Bands()

		// Define bit depth de facto, not exposed in the vips interface
		bitDepth = len(imgBytes) / (width * height * channels) * 8 // 8 bits per byte
	}

	if bitDepth == 0 && !encodeImage {
		panic("Bit depth not set")
	}

	// Release the vips image, will free underlying buffers without having to resort to the GC
	img.Close()

	imgPayload := ImagePayload{
		Data:           imgBytes,
		OriginalHeight: originalHeight,
		OriginalWidth:  originalWidth,
		Height:         height,
		Width:          width,
		Channels:       channels,
		BitDepth:       bitDepth,
	}

	return &imgPayload, aspectRatio, nil
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

func fetchImage(client *http.Client, url string, retries int, transform *ARAwareTransform, aspectRatio float64, encodeImage bool, isMask bool) (*ImagePayload, float64, error) {
	errReport := fmt.Errorf("failed fetching image %s", url)

	for i := 0; i < retries; i++ {
		// Get the raw image payload
		resp, err := client.Get(url)
		if err != nil {
			errReport = err
			exponentialBackoffWait(i)

			// Renew the client in case the connection was closed
			client = &http.Client{Timeout: 30 * time.Second}
			continue
		}
		defer resp.Body.Close()

		body_bytes, err := readBodyBuffered(resp)
		if err != nil {
			errReport = err
			exponentialBackoffWait(i)
			continue
		}

		// Decode into a flat buffer using vips
		imgPayload_ptr, aspectRatio, err := imageFromBuffer(body_bytes, transform, aspectRatio, encodeImage, isMask)
		if err != nil {
			break
		}
		return imgPayload_ptr, aspectRatio, nil
	}
	return nil, -1., errReport
}

func fetchSample(config *SourceDBConfig, httpClient *http.Client, sampleResult dbSampleMetadata, transform *ARAwareTransform, encodeImage bool) *Sample {
	// Per sample work:
	// - fetch the raw payloads
	// - deserialize / decode, depending on the types
	// return the result to the samples channel

	retries := 5
	imgPayload := &ImagePayload{}

	aspectRatio := -1. // Not initialized to begin with

	// Base image
	if config.RequireImages {
		baseImage, newAspectRatio, err := fetchImage(httpClient, sampleResult.ImageDirectURL, retries, transform, aspectRatio, encodeImage, false)

		if err != nil {
			fmt.Println("Error fetching image:", sampleResult.Id)
			return nil
		} else {
			imgPayload = baseImage
			aspectRatio = newAspectRatio
		}
	}

	// Latents
	latents := make(map[string]LatentPayload)
	masks := make(map[string]ImagePayload)
	extraImages := make(map[string]ImagePayload)

	for _, latent := range sampleResult.Latents {
		if strings.Contains(latent.LatentType, "image") && !strings.Contains(latent.LatentType, "latent_") {
			// Image types, registered as latents but they need to be jpg-decoded
			new_image, _, err := fetchImage(httpClient, latent.URL, retries, transform, aspectRatio, encodeImage, false)
			if err != nil {
				fmt.Println("Error fetching masked image:", sampleResult.Id, latent.LatentType)
				return nil
			}

			extraImages[latent.LatentType] = *new_image
		} else if latent.IsMask {
			// Mask types, registered as latents but they need to be png-decoded
			mask_ptr, _, err := fetchImage(httpClient, latent.URL, retries, transform, aspectRatio, encodeImage, true)
			if err != nil {
				fmt.Println("Error fetching mask:", sampleResult.Id, latent.LatentType)
				return nil
			}
			masks[latent.LatentType] = *mask_ptr
		} else {
			// Vanilla latents, pure binary payloads
			latentPayload, err := fetchURL(httpClient, latent.URL, retries)
			if err != nil {
				fmt.Println("Error fetching latent:", err)
				return nil
			}

			latents[latent.LatentType] = LatentPayload{
				latentPayload.content,
				len(latentPayload.content),
			}
		}
	}

	// Optional embeddings
	var cocaEmbedding []float32
	if config.RequireEmbeddings {
		cocaEmbedding = sampleResult.CocaEmbedding.Vector
	}

	return &Sample{ID: sampleResult.Id,
		Source:           sampleResult.Source,
		Attributes:       sampleResult.Attributes,
		DuplicateState:   sampleResult.DuplicateState,
		Image:            *imgPayload,
		Latents:          latents,
		Masks:            masks,
		AdditionalImages: extraImages,
		Tags:             sampleResult.Tags,
		CocaEmbedding:    cocaEmbedding}
}
