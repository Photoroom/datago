package datago

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
)

// Walk over a local directory and return the list of files
// Note that we'll page this, so that file loading can start before the full list is available

// --- File system walk structures ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type fsSampleMetadata struct {
	FilePath string `json:"file_path"`
	FileName string `json:"file_name"`
}

// -- Define the front end goroutine ---------------------------------------------------------------------------------------------------------------------------------------------------------------
type SourceFileSystemConfig struct {
	DataSourceConfig
	RootPath string `json:"root_path"`
}

func (c *SourceFileSystemConfig) setDefaults() {
	c.PageSize = 512
	c.Rank = 0
	c.WorldSize = 1

	c.RootPath = os.Getenv("DATAGO_TEST_FILESYSTEM")
}

func GetSourceFileSystemConfig() SourceFileSystemConfig {
	config := SourceFileSystemConfig{}
	config.setDefaults()
	return config
}

type datagoGeneratorFileSystem struct {
	extensions set
	config     SourceFileSystemConfig
}

func newDatagoGeneratorFileSystem(config SourceFileSystemConfig) datagoGeneratorFileSystem {
	supported_img_extensions := []string{".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
	var extensionsMap = make(set)
	for _, ext := range supported_img_extensions {
		extensionsMap.Add(ext)
	}

	if config.Rank >= config.WorldSize {
		panic("Rank should be less than World Size. Maybe you forgot to define both ?")
	}

	fmt.Println("File system root directory", config.RootPath)
	fmt.Println("Supported image extensions", supported_img_extensions)
	fmt.Println("Rank and World Size", config.Rank, config.WorldSize)

	return datagoGeneratorFileSystem{config: config, extensions: extensionsMap}
}

// hash function to distribute files across ranks
func hash(s string) int {
	h := sha256.Sum256([]byte(s))
	return int(h[0]) // Convert the first byte of the hash to an integer
}

func (f datagoGeneratorFileSystem) generatePages(ctx context.Context, chanPages *BufferedChan[Pages]) {
	// Walk over the directory and feed the results to the items channel
	// This is meant to be run in a goroutine

	var samples []SampleDataPointers

	err := filepath.Walk(f.config.RootPath, func(path string, info os.FileInfo, err error) error {
		select {
		case <-ctx.Done():
			return nil
		default:
			if err != nil {
				return err
			}

			if !info.IsDir() && f.extensions.Contains(filepath.Ext(path)) {
				if f.config.WorldSize > 1 && hash(path)%f.config.WorldSize != f.config.Rank || f.config.WorldSize == 1 {
					new_sample := fsSampleMetadata{FilePath: path, FileName: info.Name()}
					samples = append(samples, SampleDataPointers(new_sample))
				}
			}

			// Check if we have enough files to send a page
			if len(samples) >= f.config.PageSize && chanPages.open {
				chanPages.Send(Pages{samples})
				samples = nil
			}
			return nil
		}
	})

	if err != nil {
		fmt.Println("Error walking the path", f.config.RootPath)
		panic(err)
	} else {
		// Send the last page
		if len(samples) > 0 && chanPages.open {
			chanPages.Send(Pages{samples})
		}
	}
	chanPages.Close()
}
