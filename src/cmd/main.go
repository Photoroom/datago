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
	client_config.SourceType = datago.SourceTypeFileSystem
	client_config.SourceConfig = datago.GeneratorFileSystemConfig{RootPath: os.Getenv("DATAROOM_TEST_FILESYSTEM"), PageSize: 10}
	client_config.DefaultImageSize = 1024
	client_config.DownsamplingRatio = 32

	client_config.CropAndResize = *flag.Bool("crop_and_resize", false, "Whether to crop and resize the images and masks")
	client_config.ConcurrentDownloads = *flag.Int("concurrency", 64, "The number of concurrent http requests to make")
	client_config.PrefetchBufferSize = *flag.Int("item_fetch_buffer", 256, "The number of items to pre-load")
	client_config.SamplesBufferSize = *flag.Int("item_ready_buffer", 128, "The number of items ready to be served")

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
			fmt.Println("No more samples ", i, " samples served")
			break
		}
	}

	// Cancel the context to kill the goroutines
	dataroom_client.Stop()

	// Calculate the elapsed time
	elapsedTime := time.Since(startTime)
	fps := float64(*limit) / elapsedTime.Seconds()
	fmt.Printf("Total execution time: %.2f \n", elapsedTime.Seconds())
	fmt.Printf("Average throughput: %.2f samples per second\n", fps)
}
