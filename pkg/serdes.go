package datago

import (
	"bytes"
	"fmt"
	"io"
	"net/http"

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
