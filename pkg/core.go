package datago

import (
	"context"
	"errors"
	"sync/atomic"
)

type BufferedChan[T any] struct {
	_channel      chan T
	current_items int32
	max_items     int32
	open          bool
}

// --- Simple buffered channel implementation ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// Make it possible to track the current channel status from the outside

func NewBufferedChan[T any](max_items int32) BufferedChan[T] {
	return BufferedChan[T]{_channel: make(chan T, max_items), current_items: 0, max_items: max_items, open: true}
}

func (b *BufferedChan[T]) Send(item T) {
	b._channel <- item

	// Small perf hit, not sure it's worth it
	atomic.AddInt32(&b.current_items, 1)
}

func (b *BufferedChan[T]) Receive() (T, error) {
	item, open := <-b._channel
	if !open {
		return item, errors.New("Channel closed")
	}

	// Small perf hit, not sure it's worth it
	atomic.AddInt32(&b.current_items, -1)
	return item, nil
}

func (b *BufferedChan[T]) Empty() {
	consumeChannel(b._channel)
}

func (b *BufferedChan[T]) Close() {
	close(b._channel)
	b.open = false
}

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
	generatePages(ctx context.Context, chanPages *BufferedChan[Pages])
}

// The backend will be responsible for fetching the payloads and deserializing them
type Backend interface {
	collectSamples(inputSampleMetadata *BufferedChan[SampleDataPointers], outputSamples *BufferedChan[Sample], transform *ARAwareTransform, encodeImages bool)
}
