package datago

import (
	"fmt"
	"net/http"
	"time"
)

type BackendHTTP struct {
	config *DatagoConfig
}

func (b BackendHTTP) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform) {

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

			sample := fetchSample(b.config, &http_client, http_sample, transform)
			if sample != nil {
				chanSamples <- *sample
			}
		}
	}

	// Start the workers and work on the metadata channel
	for i := 0; i < b.config.ConcurrentDownloads; i++ {
		go sampleWorker()
	}

	// Wait for all the workers to be done or overall context to be cancelled
	for i := 0; i < b.config.ConcurrentDownloads; i++ {
		<-ack_channel
	}
	close(chanSamples)
	fmt.Println("No more items to serve, wrapping up")
}