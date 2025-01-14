package datago

import (
	"net/http"
	"time"
)

type BackendHTTP struct {
	config *SourceDBConfig
}

func (b BackendHTTP) collectSamples(inputSampleMetadata *BufferedChan[SampleDataPointers], outputSamples *BufferedChan[Sample], transform *ARAwareTransform, encodeImages bool) {

	sampleWorker := func(worker_handle *worker) {
		// One HHTP client per goroutine, make sure we don't run into race conditions when renewing
		http_client := http.Client{Timeout: 30 * time.Second}

		for {
			worker_handle.state = worker_idle
			item_to_fetch, open := inputSampleMetadata.Receive()
			if !open {
				worker_handle.state = worker_done
				return
			}
			worker_handle.state = worker_running

			// Cast the item to fetch to the correct type
			http_sample, ok := item_to_fetch.(dbSampleMetadata)
			if !ok {
				panic("Failed to cast the item to fetch to dbSampleMetadata. This worker is probably misconfigured")
			}

			sample := fetchSample(b.config, &http_client, http_sample, transform, encodeImages)
			if sample != nil {
				outputSamples.Send(*sample)
			}
		}
	}

	run_worker_pool(sampleWorker, inputSampleMetadata, outputSamples)
	outputSamples.Close()
}
