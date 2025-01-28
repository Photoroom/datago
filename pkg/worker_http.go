package datago

import (
	"net/http"
	"time"
)

type BackendHTTP struct {
	config *SourceDBConfig
}

func (b BackendHTTP) collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, encodeImages bool) {

	sampleWorker := func(workerHandle *worker) {
		defer workerHandle.stop()

		// One HHTP client per goroutine, make sure we don't run into race conditions when renewing
		httpClient := http.Client{Timeout: 30 * time.Second}

		for {
			workerHandle.state = workerStateIdle
			itemToFetch, open := <-chanSampleMetadata
			if !open {
				return
			}
			workerHandle.state = workerStateRunning

			// Cast the item to fetch to the correct type
			httpSample, ok := itemToFetch.(dbSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample, err := fetchSample(b.config, &httpClient, httpSample, transform, encodeImages)
			if err == nil && sample != nil {
				chanSamples <- *sample
			}
		}
	}

	defer close(chanSamples)
	runWorkerPool(sampleWorker)
}
