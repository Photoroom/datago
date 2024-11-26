package datago_test

import (
	"fmt"
	"os"
	"testing"

	datago "datago/pkg"

	"github.com/davidbyttow/govips/v2/vips"
)

func get_test_source() string {
	return os.Getenv("DATAROOM_TEST_SOURCE")
}

func get_default_test_config() datago.DatagoConfig {
	config := datago.GetDatagoConfig()

	db_config := datago.GetSourceDBConfig()
	db_config.Sources = get_test_source()
	db_config.PageSize = 32
	config.SourceConfig = db_config
	return config
}

func TestClientStartStop(t *testing.T) {
	clientConfig := get_default_test_config()

	// Check that we can start, do nothing and stop the client immediately
	client := datago.GetClient(clientConfig)
	client.Start()
	client.Stop()
}

func TestClientNoStart(t *testing.T) {
	clientConfig := get_default_test_config()

	// Check that we can get a sample without starting the client
	client := datago.GetClient(clientConfig)
	sample := client.GetSample()
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}
}

func TestClientNoStop(t *testing.T) {
	// Check that we can start, get a sample, and destroy the client immediately
	// In that case Stop() should be called in the background, and everything should work just fine
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	client := datago.GetClient(clientConfig)
	client.Start()
	_ = client.GetSample()
}

func TestMoreThanBufferSize(t *testing.T) {
	// Check that we can start, get a sample, and destroy the client immediately
	// In that case Stop() should be called in the background, and everything should work just fine
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	client := datago.GetClient(clientConfig)
	client.Start()
	_ = client.GetSample()

	if client.GetSample().ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}
	client.Stop()
}

func TestFetchImage(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	// Check that we can get an image
	client := datago.GetClient(clientConfig)
	sample := client.GetSample()

	// Assert that no error occurred
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}

	// Assert that sample is not nil or has expected properties
	if sample.Image.Height == 0 || sample.Image.Width == 0 {
		t.Errorf("Expected non-nil sample")
	}

	// Check the buffer size
	if len(sample.Image.Data) != sample.Image.Height*sample.Image.Width*3 {
		t.Errorf("Expected image buffer size to be Height*Width*3")
	}

	client.Stop()
}

func TestExtraFields(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.HasLatents = "masked_image"
	dbConfig.HasMasks = "segmentation_mask"
	clientConfig.SourceConfig = dbConfig

	// Check that we can get an image
	client := datago.GetClient(clientConfig)
	sample := client.GetSample()

	// Assert that no error occurred
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}

	// Assert that we have the expected fields in the sample
	if _, exists := sample.AdditionalImages["masked_image"]; !exists {

		t.Errorf("Sample is missing the required field %s", "masked_image")
	}

	if _, exists := sample.Masks["segmentation_mask"]; !exists {
		t.Errorf("Sample is missing the required field %s", "segmentation_mask")
	}

	client.Stop()
}

func TestCropAndResize(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1
	clientConfig.ImageConfig.CropAndResize = true

	client := datago.GetClient(clientConfig)
	client.Start()

	for i := 0; i < 10; i++ {
		sample := client.GetSample()

		// Assert that no error occurred
		if sample.ID == "" {
			t.Errorf("GetSample returned an unexpected error")
		}

		// Assert that sample is not nil or has expected properties
		if sample.Image.Height == 0 || sample.Image.Width == 0 {
			t.Errorf("Expected non-nil sample")
		}

		// Assert that image and masks are cropped and resized
		for k, v := range sample.Masks {
			if v.Height != sample.Image.Height || v.Width != sample.Image.Width {
				t.Errorf("Expected cropped and resized mask %s", k)
			}
		}

		for k, v := range sample.AdditionalImages {
			if v.Height != sample.Image.Height || v.Width != sample.Image.Width {
				t.Errorf("Expected cropped and resized image %s", k)
			}
		}
	}
	client.Stop()
}

func TestImageBufferCompression(t *testing.T) {
	// Check that the image buffer is compressed, and that we can decode it properly
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1
	clientConfig.ImageConfig.PreEncodeImages = true

	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.HasLatents = "masked_image"
	dbConfig.HasMasks = "segmentation_mask"
	clientConfig.SourceConfig = dbConfig

	client := datago.GetClient(clientConfig)
	sample := client.GetSample()

	// Check that no error occurred
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}

	// Check that the image buffers are compressed, and we can decode them properly
	// -- Test the base image
	if sample.Image.Channels != -1 {
		t.Errorf("Expected compressed image buffer")
	}

	decoded_image, err := vips.NewImageFromBuffer(sample.Image.Data)
	if err != nil {
		t.Errorf("Error decoding image buffer")
	}
	if decoded_image.Width() != sample.Image.Width || decoded_image.Height() != sample.Image.Height {
		t.Errorf("Decoded image has unexpected dimensions %d %d %d %d", decoded_image.Width(), decoded_image.Height(), sample.Image.Width, sample.Image.Height)
	}

	// -- Test the additional images (will be png compressed)
	if sample.AdditionalImages["masked_image"].Channels != -1 {
		t.Errorf("Expected compressed masked image buffer")
	}

	_, err = vips.NewImageFromBuffer(sample.AdditionalImages["masked_image"].Data)
	if err != nil {
		t.Errorf("Error decoding masked image buffer")
	}

	// -- Test the masks (will be png compressed)
	if sample.Masks["segmentation_mask"].Channels != -1 {
		t.Errorf("Expected compressed mask buffer")
	}

	_, err = vips.NewImageFromBuffer(sample.Masks["segmentation_mask"].Data)
	if err != nil {
		t.Errorf("Error decoding mask buffer")
	}
	client.Stop()
}

func TestStrings(t *testing.T) {
	clientConfig := get_default_test_config()
	client := datago.GetClient(clientConfig)
	client.Start()

	for i := 0; i < 10; i++ {
		sample := client.GetSample()

		// Assert that no error occurred
		if sample.ID == "" {
			t.Errorf("GetSample returned an unexpected error")
		}

		// Check that we can decode all the strings
		if string(sample.ID) == "" {
			t.Errorf("Expected non-empty string")
		}

		fmt.Println(string(sample.ID))
	}
	client.Stop()
}

func TestRanks(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.WorldSize = 2
	dbConfig.Rank = 0
	dbConfig.RequireImages = false
	clientConfig.SourceConfig = dbConfig

	client_0 := datago.GetClient(clientConfig)

	dbConfig.Rank = 1
	clientConfig.SourceConfig = dbConfig
	client_1 := datago.GetClient(clientConfig)

	client_0.Start()
	client_1.Start()

	samples_0 := make(map[string]int)
	samples_1 := make(map[string]int)

	for i := 0; i < 10; i++ {
		sample_0 := client_0.GetSample()
		sample_1 := client_1.GetSample()

		if sample_0.ID == "" || sample_1.ID == "" {
			t.Errorf("GetSample returned an unexpected error")
		}

		samples_0[sample_0.ID] = 1
		samples_1[sample_1.ID] = 1

	}

	// Check that there are no keys in common in between the two samples
	for k := range samples_0 {
		if _, exists := samples_1[k]; exists {
			t.Errorf("Samples are not distributed across ranks")
		}
	}

	client_0.Stop()
	client_1.Stop()
}

func TestTags(t *testing.T) {
	clientConfig := get_default_test_config()
	{
		clientConfig.SamplesBufferSize = 1

		dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
		dbConfig.Tags = "v4_trainset_hq"
		dbConfig.RequireImages = false
		clientConfig.SourceConfig = dbConfig

		client := datago.GetClient(clientConfig)

		countains := func(slice []string, element string) bool {
			for _, item := range slice {
				if item == element {
					return true
				}
			}
			return false
		}

		for i := 0; i < 10; i++ {
			sample := client.GetSample()
			if !countains(sample.Tags, "v4_trainset_hq") {
				t.Errorf("Sample is missing the required tag")
			}
		}
		client.Stop()
	}
	{
		clientConfig.SamplesBufferSize = 1

		dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
		dbConfig.Tags = ""
		dbConfig.TagsNE = "v4_trainset_hq"
		dbConfig.RequireImages = false

		clientConfig.SourceConfig = dbConfig

		client := datago.GetClient(clientConfig)

		countains := func(slice []string, element string) bool {
			for _, item := range slice {
				if item == element {
					return true
				}
			}
			return false
		}

		for i := 0; i < 10; i++ {
			sample := client.GetSample()
			if countains(sample.Tags, "v4_trainset_hq") {
				t.Errorf("Sample is missing the required tag")
			}
		}
		client.Stop()
	}
}

func TestMultipleSources(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.Sources = "LAION_ART,LAION_AESTHETICS"
	dbConfig.RequireImages = false
	clientConfig.SourceConfig = dbConfig

	client := datago.GetClient(clientConfig)

	// Pull samples from the client, collect the sources
	test_set := make(map[string]interface{})
	for i := 0; i < 100; i++ {
		sample := client.GetSample()
		if _, exists := test_set[sample.Source]; !exists {
			test_set[sample.Source] = nil
			if len(test_set) == 2 {
				break
			}
		}
	}

	isin := func(dict map[string]interface{}, element string) bool {
		_, exists := dict[element]
		return exists
	}

	if len(test_set) != 2 || !isin(test_set, "LAION_ART") || !isin(test_set, "LAION_AESTHETICS") {
		t.Error("Missing the required sources")
		fmt.Println(test_set)
	}
	client.Stop()
}

func TestSourcesNE(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1

	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.Sources = "LAION_ART,LAION_AESTHETICS"
	dbConfig.SourcesNE = "LAION_ART"
	dbConfig.RequireImages = false
	clientConfig.SourceConfig = dbConfig

	client := datago.GetClient(clientConfig)

	// Pull samples from the client, collect the sources
	for i := 0; i < 100; i++ {
		sample := client.GetSample()
		if sample.Source == "LAION_ART" {
			t.Error("We're not supposed to get samples from LAION_ART")
		}
	}
	client.Stop()
}

func TestRandomSampling(t *testing.T) {
	clientConfig := get_default_test_config()
	clientConfig.SamplesBufferSize = 1
	dbConfig := clientConfig.SourceConfig.(datago.SourceDBConfig)
	dbConfig.RandomSampling = true
	clientConfig.SourceConfig = dbConfig

	// Fill in two sets with some results
	sample_set_1 := make(map[string]interface{})
	sample_set_2 := make(map[string]interface{})

	{
		client := datago.GetClient(clientConfig)

		for i := 0; i < 10; i++ {
			sample := client.GetSample()
			sample_set_1[sample.ID] = nil
		}
		client.Stop()
	}

	{
		client := datago.GetClient(clientConfig)

		for i := 0; i < 10; i++ {
			sample := client.GetSample()
			sample_set_2[sample.ID] = nil
		}
		client.Stop()
	}

	// Check that the two sets are different
	setsAreEqual := func(set1, set2 map[string]interface{}) bool {
		if len(set1) != len(set2) {
			return false
		}
		for k := range set1 {
			if _, exists := set2[k]; !exists {
				return false
			}
		}
		return true
	}

	if setsAreEqual(sample_set_1, sample_set_2) {
		t.Error("Random sampling is not working")
	}
}
