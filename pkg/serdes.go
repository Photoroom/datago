package datago

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
)

func sanitizeImage(img *vips.ImageRef, isMask bool) error {

	// Catch possible crash in libvips and recover from it
	defer func() error {
		if r := recover(); r != nil {
			return fmt.Errorf("caught crash: %v", r)
		}
		return nil
	}()

	// If the image is 4 channels, we need to drop the alpha channel
	if img.Bands() == 4 {
		err := img.Flatten(&vips.Color{R: 255, G: 255, B: 255}) // Flatten with white background
		if err != nil {
			return fmt.Errorf("error flattening image: %w", err)
		}
		fmt.Println("Image flattened")
	}

	// If the image is not a mask but is 1 channel, we want to convert it to 3 channels
	if (img.Bands() == 1) && !isMask {
		if img.Metadata().Format == vips.ImageTypeJPEG {
			err := img.ToColorSpace(vips.InterpretationSRGB)
			if err != nil {
				return fmt.Errorf("error converting to sRGB: %w", err)
			}
		} else {
			// // FIXME: maybe that we could recover these still. By default throws an error, sRGB and PNG not supported
			return fmt.Errorf("1 channel PNG image not supported")
		}
	}

	// If the image is 2 channels, that's gray+alpha and we flatten it
	if img.Bands() == 2 {
		err := img.ExtractBand(1, 1)
		fmt.Println("Gray+alpha image, removing alpha")
		if err != nil {
			return fmt.Errorf("error extracting band: %w", err)
		}
	}

	return nil
}

func imageFromBuffer(buffer []byte, transform *ARAwareTransform, aspectRatio float64, encodeImage bool, isMask bool) (*ImagePayload, float64, error) {
	// Decode the image payload using vips, using bulletproof settings
	importParams := vips.NewImportParams()
	importParams.AutoRotate.Set(true)
	importParams.FailOnError.Set(true)
	importParams.Page.Set(0)
	importParams.NumPages.Set(1)
	importParams.HeifThumbnail.Set(false)
	importParams.SvgUnlimited.Set(false)

	img, err := vips.LoadImageFromBuffer(buffer, importParams)
	if err != nil {
		return nil, -1., fmt.Errorf("error loading image: %w", err)
	}

	err = sanitizeImage(img, isMask)
	if err != nil {
		return nil, -1., fmt.Errorf("error processing image: %w", err)
	}

	// Optionally crop and resize the image on the fly. Save the aspect ratio in the process for future use
	originalWidth, originalHeight := img.Width(), img.Height()

	if transform != nil {
		aspectRatio, err = transform.cropAndResizeToClosestAspectRatio(img, aspectRatio)
		if err != nil {
			return nil, -1., fmt.Errorf("error cropping and resizing image: %w", err)
		}
	}

	width, height := img.Width(), img.Height()

	// If requested, re-encode the image to a jpg or png
	var imgBytes []byte
	var channels int
	var bitDepth int

	if encodeImage {
		if img.Bands() == 3 {
			// Re-encode the image to a jpg
			imgBytes, _, err = img.ExportJpeg(&vips.JpegExportParams{Quality: 95})
		} else {
			// Re-encode the image to a png
			imgBytes, _, err = img.ExportPng(&vips.PngExportParams{
				Compression: 6,
				Filter:      vips.PngFilterNone,
				Interlace:   false,
				Palette:     false,
				Bitdepth:    8, // force 8 bit depth
			})
		}

		if err != nil {
			return nil, -1., err
		}

		channels = -1 // Signal that we have encoded the image
	} else {
		channels = img.Bands()
		imgBytes, err = img.ToBytes()

		if err != nil {
			return nil, -1., err
		}
		// Define bit depth de facto, not exposed in the vips interface
		bitDepth = len(imgBytes) / (width * height * channels) * 8 // 8 bits per byte
	}

	defer img.Close() // release vips buffers when done

	if bitDepth == 0 && !encodeImage {
		panic("Bit depth not set")
	}

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

		body_bytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()

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

		body_bytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if err != nil {
			errReport = err
			exponentialBackoffWait(i)
			continue
		}

		// Decode into a flat buffer using vips. Note that this can fail on its own
		{
			imgPayload_ptr, aspectRatio, err := imageFromBuffer(body_bytes, transform, aspectRatio, encodeImage, isMask)
			if err != nil {
				errReport = err
				continue
			}
			return imgPayload_ptr, aspectRatio, nil
		}
	}
	return nil, -1., errReport
}

func fetchSample(config *SourceDBConfig, httpClient *http.Client, sampleResult dbSampleMetadata, transform *ARAwareTransform, encodeImage bool) (*Sample, error) {
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
			return nil, fmt.Errorf("Error fetching image:", sampleResult.Id)
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
				return nil, fmt.Errorf("Error fetching masked image:", sampleResult.Id, latent.LatentType)
			}

			extraImages[latent.LatentType] = *new_image
		} else if latent.IsMask {
			// Mask types, registered as latents but they need to be png-decoded
			mask_ptr, _, err := fetchImage(httpClient, latent.URL, retries, transform, aspectRatio, encodeImage, true)
			if err != nil {
				return nil, fmt.Errorf("Error fetching mask:", sampleResult.Id, latent.LatentType)
			}
			masks[latent.LatentType] = *mask_ptr
		} else {
			fmt.Println("Loading latents ", latent.URL)

			// Vanilla latents, pure binary payloads
			latentPayload, err := fetchURL(httpClient, latent.URL, retries)
			if err != nil {
				return nil, fmt.Errorf("Error fetching latent:", err)
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
		CocaEmbedding:    cocaEmbedding}, nil
}
