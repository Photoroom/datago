package datago

import (
	"time"
)

func exponentialBackoffWait(retries int) {
	baseDelay := time.Second
	maxDelay := 64 * time.Second

	// Calculate the delay with exponential backoff
	delay := baseDelay * (1 << uint(retries))
	if delay > maxDelay {
		delay = maxDelay
	}
	time.Sleep(delay)
}

func getLast5Chars(s string) string {
	runes := []rune(s)
	if len(runes) <= 5 {
		return s
	}
	return string(runes[len(runes)-5:])
}

func consumeChannel[T any](ch <-chan T) {
	for range ch {
	}
}

// Define a set type using a map with empty struct values
type set map[string]struct{}

// Add an element to the set
func (s set) Add(value string) {
	s[value] = struct{}{}
}

// Remove an element from the set
func (s set) Remove(value string) {
	delete(s, value)
}

// Check if the set contains an element
func (s set) Contains(value string) bool {
	_, exists := s[value]
	return exists
}

// Get the size of the set
func (s set) Size() int {
	return len(s)
}
