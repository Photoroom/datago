package datago

import (
	"fmt"

	"golang.org/x/exp/mmap"
)

type BackendFileSystem struct {
	config *DatagoConfig
}

func loadSample(filesystem_sample fsSampleMetadata, transform *ARAwareTransform, pre_encode_image bool) *Sample {
	// Using mmap to put the file directly into memory, removes buffering needs
	r, err := mmap.Open(filesystem_sample.FilePath)
	if err != nil {
		panic(err)
	}

	bytes_buffer := make([]byte, r.Len())
	_, err = r.ReadAt(bytes_buffer, 0)
	if err != nil {
		panic(err)
	}

	// Decode the image, can error out here also, and return the sample
	img_payload, _, err := imageFromBuffer(bytes_buffer, transform, -1., pre_encode_image, false)
	if err != nil {
		fmt.Println("Error loading image:", filesystem_sample.FileName)
		return nil
	}

	return &Sample{ID: filesystem_sample.FileName,
		Image: *img_payload,
	}
}

func (b BackendFileSystem) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, encodeImages bool) {

	sampleWorker := func(worker_handle *worker) {
		defer worker_handle.stop()

		for {
			worker_handle.state = worker_idle
			item_to_fetch, open := <-chanSampleMetadata

			if !open {
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
				chanSamples <- *sample
			}
		}
	}

	defer close(chanSamples)
	runWorkerPool(sampleWorker)
}
