package datago

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
)

// Walk over a webdataset target, pull the tar files
// Note that we'll page this, so that file loading can start from the get go

// --- File system walk structures ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type webSampleMetadata struct {
}

// -- Define the front end goroutine ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type SourceWebDatasetConfig struct {
	DataSourceConfig
	RootPath string `json:"root_path"`
}

func (c *SourceWebDatasetConfig) setDefaults() {
	c.PageSize = 512
	c.Rank = 0
	c.WorldSize = 1

	c.RootPath = os.Getenv("DATAGO_TEST_WEBDATASET")
}

func GetSourceFileSystemConfig() SourceWebDatasetConfig {
	config := SourceWebDatasetConfig{}
	config.setDefaults()
	return config
}

type datagoGeneratorWebDataset struct {
	extensions set
	config     SourceWebDatasetConfig
}

func newDatagoGeneratorWebDataset(config SourceWebDatasetConfig) datagoGeneratorWebDataset {
	var extensionsMap = make(set)
	for _, ext := range imgExtensions {
		extensionsMap.Add(ext)
	}

	if config.Rank >= config.WorldSize {
		panic("Rank should be less than World Size. Maybe you forgot to define both ?")
	}

	fmt.Println("File system root directory", config.RootPath)
	fmt.Println("Supported image extensions", imgExtensions)
	fmt.Println("Rank and World Size", config.Rank, config.WorldSize)

	return datagoGeneratorWebDataset{config: config, extensions: extensionsMap}
}

// hash function to distribute files across ranks
func hash(s string) int {
	h := sha256.Sum256([]byte(s))
	return int(h[0]) // Convert the first byte of the hash to an integer
}

func (f datagoGeneratorWebDataset) generatePages(ctx context.Context, chanPages chan Pages) {
	// Walk over the directory and feed the results to the items channel
	// This is meant to be run in a goroutine

	var samples []SampleDataPointers

	// Something related needs to be written with respect to the tar files,
	// while grouping the related files together

	// Decoding will not be here, we just move the paths down the line

	// err := filepath.Walk(f.config.RootPath, func(path string, info os.FileInfo, err error) error {
	// 	if err != nil {
	// 		return err
	// 	}

	// 	if !info.IsDir() && f.extensions.Contains(filepath.Ext(path)) {
	// 		if f.config.WorldSize > 1 && hash(path)%f.config.WorldSize != f.config.Rank || f.config.WorldSize == 1 {
	// 			new_sample := fsSampleMetadata{FilePath: path, FileName: info.Name()}
	// 			samples = append(samples, SampleDataPointers(new_sample))
	// 		}
	// 	}

	// 	// Check if we have enough files to send a page
	// 	if len(samples) >= f.config.PageSize {
	// 		chanPages <- Pages{samples}
	// 		samples = nil
	// 	}
	// 	return nil
	// })

	// if err != nil {
	// 	fmt.Println("Error walking the path", f.config.RootPath)
	// 	panic(err)
	// } else {
	// 	// Send the last page
	// 	if len(samples) > 0 {
	// 		chanPages <- Pages{samples}
	// 	}
	// }

	close(chanPages)
}
