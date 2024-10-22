package datago

import (
	"unsafe"
)

func dataPtrFromSlice(a []uint8) uintptr {
	if len(a) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&a[0]))
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
