package main

import (
	datago "datago/pkg"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"runtime/trace"
	"time"
)

func main() {
	// Define flags
	client_config := datago.GetDefaultConfig()
	client_config.DefaultImageSize = 1024
	client_config.DownsamplingRatio = 32

	client_config.CropAndResize = *flag.Bool("crop_and_resize", false, "Whether to crop and resize the images and masks")
	client_config.ConcurrentDownloads = *flag.Int("concurrency", 64, "The number of concurrent http requests to make")
	client_config.PrefetchBufferSize = *flag.Int("item_fetch_buffer", 256, "The number of items to pre-load")
	client_config.SamplesBufferSize = *flag.Int("item_ready_buffer", 128, "The number of items ready to be served")

	client_config.Sources = *flag.String("source", "GETTY", "The source for the items")
	client_config.RequireImages = *flag.Bool("require_images", true, "Whether the items require images")
	client_config.RequireEmbeddings = *flag.Bool("require_embeddings", false, "Whether the items require the DB embeddings")

	client_config.Tags = *flag.String("tags", "", "The tags to filter for")
	client_config.TagsNE = *flag.String("tags__ne", "", "The tags that the samples should not have")
	client_config.HasMasks = *flag.String("has_masks", "", "The masks to filter for")
	client_config.HasLatents = *flag.String("has_latents", "", "The masks to filter for")
	client_config.HasAttributes = *flag.String("has_attributes", "", "The attributes to filter for")

	client_config.LacksMasks = *flag.String("lacks_masks", "", "The masks to filter against")
	client_config.LacksLatents = *flag.String("lacks_latents", "", "The masks to filter against")
	client_config.LacksAttributes = *flag.String("lacks_attributes", "", "The attributes to filter against")

	limit := flag.Int("limit", 2000, "The number of items to fetch")
	profile := flag.Bool("profile", false, "Whether to profile the code")

	// Parse the flags and instantiate the client
	flag.Parse()

	dataroom_client := datago.GetClient(client_config)

	// Go-routine which will feed the sample data to the workers
	// and fetch the next page
	startTime := time.Now() // Record the start time

	if *profile {
		fmt.Println("Profiling the code")
		{
			f, _ := os.Create("trace.out")
			// read with go tool trace trace.out

			err := trace.Start(f)
			if err != nil {
				panic(err)
			}
			defer trace.Stop()
		}
		{
			f, _ := os.Create("cpu.prof")
			// read with go tool pprof cpu.prof
			err := pprof.StartCPUProfile(f)
			if err != nil {
				panic(err)
			}
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
