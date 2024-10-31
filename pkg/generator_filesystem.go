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

type GeneratorFileSystemConfig struct {
	RootPath string
	PageSize int
}

func newDatagoGeneratorFileSystem(config GeneratorFileSystemConfig) datagoGeneratorFileSystem {
	supported_img_extensions := []string{".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
	var extensionsMap = make(set)
	for _, ext := range supported_img_extensions {
		extensionsMap.Add(ext)
	}
	fmt.Println("File system root directory", config.RootPath)
	fmt.Println("Supported image extensions", supported_img_extensions)

	return datagoGeneratorFileSystem{root_directory: config.RootPath, extensions: extensionsMap, page_size: config.PageSize}
}

func (f datagoGeneratorFileSystem) generatePages(ctx context.Context, chanPages chan Pages) {
	// Walk over the directory and feed the results to the items channel
	// This is meant to be run in a goroutine

	var samples []SampleDataPointers

	err := filepath.Walk(f.root_directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && f.extensions.Contains(filepath.Ext(path)) {
			new_sample := fsSampleMetadata{filePath: path, fileName: info.Name()}
			samples = append(samples, SampleDataPointers(new_sample))
		}

		// Check if we have enough files to send a page
		if len(samples) >= f.page_size {
			chanPages <- Pages{samples}
			samples = nil
		}
		return nil
	})

	if err != nil {
		fmt.Println("Error walking the path", f.root_directory)
	} else {
		// Send the last page
		if len(samples) > 0 {
			chanPages <- Pages{samples}
		}
	}

	close(chanPages)
}
