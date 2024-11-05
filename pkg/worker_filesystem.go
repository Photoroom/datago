package datago

import (
	"fmt"
	"os"
)

type BackendFileSystem struct {
	config      *DatagoConfig
	concurrency int
}

func loadSample(config *DatagoConfig, filesystem_sample fsSampleMetadata, transform *ARAwareTransform, pre_encode_images bool) *Sample {
	// Load the file into []bytes
	bytes_buffer, err := os.ReadFile(filesystem_sample.FilePath)
	if err != nil {
		fmt.Println("Error reading file:", filesystem_sample.FilePath)
		return nil
	}

	img_payload, _, err := imageFromBuffer(bytes_buffer, transform, -1., pre_encode_images, false)
	if err != nil {
		fmt.Println("Error loading image:", filesystem_sample.FileName)
		return nil
	}

	return &Sample{ID: filesystem_sample.FileName,
		Image: *img_payload,
	}
}

func (b BackendFileSystem) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, pre_encode_images bool) {

	ack_channel := make(chan bool)

	sampleWorker := func() {
		for {
			item_to_fetch, open := <-chanSampleMetadata
			if !open {
				ack_channel <- true
				return
			}

			// Cast the item to fetch to the correct type
			filesystem_sample, ok := item_to_fetch.(fsSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample := loadSample(b.config, filesystem_sample, transform, pre_encode_images)
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
	fmt.Println("No more items to serve, wrapping up")
}
