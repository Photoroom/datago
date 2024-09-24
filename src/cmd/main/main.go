package main

import (
	datago "datago/pkg/client"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"runtime/trace"
	"time"
)

func main() {
	// Define flags
	require_images := flag.Bool("require_images", true, "Whether the items require images")
	require_embeddings := flag.Bool("require_embeddings", false, "Whether the items require the DB embeddings")

	crop_and_resize := flag.Bool("crop_and_resize", false, "Whether to crop and resize the images and masks")

	concurrency := flag.Int("concurrency", 64, "The number of concurrent http requests to make")
	item_fetch_buffer := flag.Int("item_fetch_buffer", 256, "The number of items to pre-load")
	item_ready_buffer := flag.Int("item_ready_buffer", 128, "The number of items ready to be served")
	limit := flag.Int("limit", 2000, "The number of items to fetch")
	source := flag.String("source", "SOURCE", "The source for the items")

	tags := flag.String("tags", "", "The tags to filter for")
	tags__ne := flag.String("tags__ne", "", "The tags used for inversed filtering")

	has_masks := flag.String("has_masks", "", "The masks to filter for")
	has_latents := flag.String("has_latents", "", "The masks to filter for")
	has_attributes := flag.String("has_attributes", "", "The attributes to filter for")

	lacks_masks := flag.String("lacks_masks", "", "The masks to filter against")
	lacks_latents := flag.String("lacks_latents", "", "The masks to filter against")
	lacks_attributes := flag.String("lacks_attributes", "", "The attributes to filter against")

	profile := flag.Bool("profile", false, "Whether to profile the code")

	// Parse the flags and instantiate the client
	flag.Parse()
	rank := uint32(0)
	world_size := uint32(1)

	dataroom_client := datago.GetClient(
		*source,
		*require_images,
		*require_embeddings,
		*tags, *tags__ne,
		*has_attributes, *lacks_attributes,
		*has_masks, *lacks_masks,
		*has_latents, *lacks_latents,
		*crop_and_resize,
		1024, 32, false,
		rank, world_size,
		*item_fetch_buffer,
		*item_ready_buffer,
		*concurrency,
	)

	// Go-routine which will feed the sample data to the workers
	// and fetch the next page
	startTime := time.Now() // Record the start time

	if *profile {
		fmt.Println("Profiling the code")
		{
			f, _ := os.Create("trace.out")
			// read with go tool trace trace.out

			trace.Start(f)
			defer trace.Stop()
		}
		{
			f, _ := os.Create("cpu.prof")
			// read with go tool pprof cpu.prof
			pprof.StartCPUProfile(f)
			defer pprof.StopCPUProfile()
		}
	}

	dataroom_client.Start()

	// Fetch all of the binary payloads as they become available
	// NOTE: This is useless, just making sure that we empty the payloads channel
	for i := 0; i < *limit; i++ {
		sample := dataroom_client.GetSample()
		if sample.ID == "" {
			fmt.Printf("Error fetching sample")
			break
		}
	}

	// Cancel the context to kill the goroutines
	dataroom_client.Stop()

	// Calculate the elapsed time
	elapsedTime := time.Since(startTime)
	fps := float64(*limit) / elapsedTime.Seconds()
	fmt.Printf("Total execution time: %.2f \n", elapsedTime.Seconds())
	fmt.Printf("Average fetch rate: %.2f fetches per second\n", fps)
}
