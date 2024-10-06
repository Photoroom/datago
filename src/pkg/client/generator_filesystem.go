package datago

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// Walk over a local directory and return the list of files
// Note that we'll page this, so that file loading can start before the full list is available

// --- File system walk structures ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type fsSampleMetadata struct {
	filePath string
	fileName string
}

// -- Define the front end goroutine ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type datagoGeneratorFileSystem struct {
	root_directory string
	extensions     set
	page_size      int
}

func newDatagoGeneratorFileSystem(config DatagoConfig) datagoGeneratorFileSystem {
	supported_img_extensions := []string{"jpg", "jpeg", "png"}
	var extensionsMap = make(set)
	for _, ext := range supported_img_extensions {
		extensionsMap.Add(ext)
	}
	fmt.Println("File system root directory", config.Sources)
	fmt.Println("Supported image extensions", supported_img_extensions)

	return datagoGeneratorFileSystem{root_directory: config.Sources, extensions: extensionsMap, page_size: config.PageSize}
}

func (f datagoGeneratorFileSystem) generatePages(ctx context.Context, chanPages chan Pages) {
	// Walk over the directory and feed the results to the items channel
	// This is meant to be run in a goroutine

	var samples []fsSampleMetadata

	_ = filepath.Walk(f.root_directory, func(path string, info os.FileInfo, err error) error {
		fmt.Println("Walking", path) // DEBUG
		if err != nil {
			return err
		}
		if !info.IsDir() && f.extensions.Contains(filepath.Ext(path)) {
			fmt.Println("Adding", path) // DEBUG
			samples = append(samples, fsSampleMetadata{filePath: path, fileName: info.Name()})
		}

		// Check if we have enough files to send a page
		if len(samples) >= f.page_size {
			// TODO: There's probably a better way to do this, we re doing useless copies here
			samplesDataPointers := make([]SampleDataPointers, len(samples))

			for i, sample := range samples {
				samplesDataPointers[i] = sample
			}

			chanPages <- Pages{samplesDataPointers}
			samples = nil
		}
		return nil
	})

}
