package datago

import (
	"fmt"
	"os"
)

type BackendFileSystem struct {
	config *DatagoConfig
}

func loadFromDisk(fsSample fsSampleMetadata, transform *ARAwareTransform, encodeImage bool) (*Sample, error) {
	// Load the file into []bytes
	bytesBuffer, err := os.ReadFile(fsSample.FilePath)
	if err != nil {
		return nil, fmt.Errorf("Error reading file:", fsSample.FilePath)
	}

	// Slightly faster take, requires Go 1.21+ which breaks gopy speed for now
	{
		// // Using mmap to put the file directly into memory, removes buffering needs
		// r, err := mmap.Open(fsSample.FilePath)
		// if err != nil {
		// 	panic(err)
		// }

		// bytesBuffer := make([]byte, r.Len())
		// _, err = r.ReadAt(bytesBuffer, 0)
		// if err != nil {
		// 	panic(err)
		// }
	}

	// Decode the image, can error out here also, and return the sample
	imgPayload, _, err := imageFromBuffer(bytesBuffer, transform, -1., encodeImage, false)
	if err != nil {
		return nil, fmt.Errorf("Error loading image:", fsSample.FileName)
	}

	return &Sample{ID: fsSample.FileName,
		Image: *imgPayload,
	}, nil
}

func (b BackendFileSystem) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, encodeImages bool) {

	sampleWorker := func(workerHandle *worker) {
		defer workerHandle.stop()

		for {
			workerHandle.state = workerStateIdle
			item_to_fetch, open := <-chanSampleMetadata

			if !open {
				return
			}
			workerHandle.state = workerStateRunning

			// Cast the item to fetch to the correct type
			fsSample, ok := item_to_fetch.(fsSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample, err := loadFromDisk(fsSample, transform, encodeImages)
			if err == nil && sample != nil {
				chanSamples <- *sample
			}
		}
	}

	defer close(chanSamples)
	runWorkerPool(sampleWorker)
}
