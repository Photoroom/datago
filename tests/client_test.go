package datago_test

import (
	"os"
	"testing"

	datago "datago/pkg"

	"github.com/davidbyttow/govips/v2/vips"
)

func get_test_source() string {
	return os.Getenv("DATAROOM_TEST_SOURCE")
}

func get_default_test_config() datago.DatagoConfig {
	config := datago.DatagoConfig{}
	config.SetDefaults()

	db_config := datago.GeneratorDBConfig{}
	db_config.SetDefaults()
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

	dbConfig := clientConfig.SourceConfig.(datago.GeneratorDBConfig)
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

	dbConfig := clientConfig.SourceConfig.(datago.GeneratorDBConfig)
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
}
