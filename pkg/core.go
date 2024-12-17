package datago

import "context"

// --- Sample data structures - these will be exposed to the Python world ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type LatentPayload struct {
	Data []byte
	Len  int
}

type ImagePayload struct {
	Data           []byte
	OriginalHeight int // Good indicator of the image frequency dbResponse at the current resolution
	OriginalWidth  int
	Height         int // Useful to decode the current payload
	Width          int
	Channels       int
	BitDepth       int
}

type Sample struct {
	ID               string
	Source           string
	Attributes       map[string]interface{}
	DuplicateState   int
	Image            ImagePayload
	Masks            map[string]ImagePayload
	AdditionalImages map[string]ImagePayload
	Latents          map[string]LatentPayload
	CocaEmbedding    []float32
	Tags             []string
}

// --- Generator and Backend interfaces ---------------------------------------------------------------------------------------------------------------------------------------------------------------

// The generator will be responsible for producing pages of metadata which can be dispatched
// to the dispatch goroutine. The metadata will be used to fetch the actual payloads

type SampleDataPointers interface{}

type Pages struct {
	samplesDataPointers []SampleDataPointers
}

type Generator interface {
	generatePages(ctx context.Context, chanPages chan Pages)
}

// The backend will be responsible for fetching the payloads and deserializing them
type Backend interface {
	collectSamples(chanSampleMetadata chan SampleDataPointers, chanSamples chan Sample, transform *ARAwareTransform, pre_encode_images bool)
}
