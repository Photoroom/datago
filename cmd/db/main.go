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

	cropAndResize := flag.Bool("crop_and_resize", false, "Whether to crop and resize the images and masks")
	itemFetchBuffer := flag.Int("item_fetch_buffer", 256, "The number of items to pre-load")
	itemReadyBuffer := flag.Int("item_ready_buffer", 128, "The number of items ready to be served")
	limit := flag.Int("limit", 2000, "The number of items to fetch")
	profile := flag.Bool("profile", false, "Whether to profile the code")
	source := flag.String("source", os.Getenv("DATAGO_TEST_DB"), "The data source to select on")

	// Parse the flags before setting the configuration values
	flag.Parse()

	// Initialize the configuration
	config := datago.GetDatagoConfig()

	sourceConfig := datago.GetDefaultSourceDBConfig()
	sourceConfig.Sources = *source

	config.ImageConfig = datago.GetDefaultImageTransformConfig()
	config.ImageConfig.CropAndResize = *cropAndResize

	config.SourceConfig = sourceConfig
	config.PrefetchBufferSize = int32(*itemFetchBuffer)
	config.SamplesBufferSize = int32(*itemReadyBuffer)
	config.Limit = *limit

	dataroom_client := datago.GetClient(config)

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
	n_samples := 0
	for {
		sample := dataroom_client.GetSample()
		if sample.ID == "" {
			fmt.Println("No more samples")
			break
		}
		n_samples++
	}

	// Cancel the context to kill the goroutines
	dataroom_client.Stop()

	// Calculate the elapsed time
	elapsedTime := time.Since(startTime)
	fps := float64(config.Limit) / elapsedTime.Seconds()
	fmt.Printf("Total execution time: %.2f seconds. Samples %d \n", elapsedTime.Seconds(), n_samples)
	fmt.Printf("Average throughput: %.2f samples per second \n", fps)
}
