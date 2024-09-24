package datago_test

import (
	"testing"

	datago "datago/pkg/client"
)

func TestClientStartStop(t *testing.T) {
	// Check that we can start, do nothing and stop the client immediately
	client := datago.GetClient("SOURCE", true, false, "", "", "", "", "", "", "", "", false, 1024, 16, false, 0, 1, 8, 8, 2)
	client.Start()
	client.Stop()
}

func TestClientNoStart(t *testing.T) {
	// Check that we can start, do nothing and stop the client immediately
	client := datago.GetClient("SOURCE", true, false, "", "", "", "", "", "", "", "", false, 1024, 16, false, 0, 1, 8, 8, 2)
	sample := client.GetSample()
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}
}

func TestClientNoStop(t *testing.T) {
	// Check that we can start, do nothing and destroy the client immediately
	// In that case Stop() should be called in the background, and everything should work just fine
	client := datago.GetClient("SOURCE", true, false, "", "", "", "", "", "", "", "", false, 1024, 16, false, 0, 1, 8, 8, 2)
	client.Start()
	_ = client.GetSample()

}

func TestFetchImage(t *testing.T) {
	client := datago.GetClient("SOURCE", true, false, "", "", "", "", "", "", "", "", false, 1024, 16, false, 0, 1, 8, 8, 2)
	client.Start()
	sample := client.GetSample()

	// Assert that no error occurred
	if sample.ID == "" {
		t.Errorf("GetSample returned an unexpected error")
	}

	// Assert that sample is not nil or has expected properties
	if sample.Image.Height == 0 || sample.Image.Width == 0 {
		t.Errorf("Expected non-nil sample")
	}

	client.Stop()
}

func TestCropAndResize(t *testing.T) {
	client := datago.GetClient("SOURCE", true, false, "", "", "", "", "segmentation_mask", "", "masked_image", "", true, 1024, 16, false, 0, 1, 8, 8, 2)
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
	}
	client.Stop()
}

// TODO: ask for something which doesn't exist
