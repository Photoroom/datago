package datago

import (
	"fmt"
	"os"
)

type BackendFileSystem struct {
	config *DatagoConfig
}

func loadSample(filesystem_sample fsSampleMetadata, transform *ARAwareTransform, pre_encode_image bool) *Sample {
	// Load the file into []bytes
	bytes_buffer, err := os.ReadFile(filesystem_sample.FilePath)
	if err != nil {
		fmt.Println("Error reading file:", filesystem_sample.FilePath)
		return nil
	}

	img_payload, _, err := imageFromBuffer(bytes_buffer, transform, -1., pre_encode_image, false)
	if err != nil {
		fmt.Println("Error loading image:", filesystem_sample.FileName)
		return nil
	}

	return &Sample{ID: filesystem_sample.FileName,
		Image: *img_payload,
	}
}

func (b BackendFileSystem) collectSamples(inputSampleMetadata *BufferedChan[SampleDataPointers], outputSamples *BufferedChan[Sample], transform *ARAwareTransform, encodeImages bool) {

	sampleWorker := func(worker_handle *worker) {
		for {
			if worker_handle.state == worker_stopping {
				worker_handle.state = worker_done
				return
			}
			worker_handle.state = worker_idle
			item_to_fetch, open := inputSampleMetadata.Receive()
			if !open {
				worker_handle.state = worker_done
				return
			}
			worker_handle.state = worker_running

			// Cast the item to fetch to the correct type
			filesystem_sample, ok := item_to_fetch.(fsSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample := loadSample(filesystem_sample, transform, encodeImages)
			if sample != nil {
				outputSamples.Send(*sample)
			}
		}
	}

	run_worker_pool(sampleWorker, inputSampleMetadata, outputSamples)
	outputSamples.Close()
}
