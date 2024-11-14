package datago_test

import (
	datago "datago/pkg"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"testing"
	"time"
)

// Function to generate and save a random image
func generateRandomImage(width, height int, filename string) error {
	// Create a new RGBA image
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Fill the image with random colors
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8(rand.Intn(256))
			g := uint8(rand.Intn(256))
			b := uint8(rand.Intn(256))
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	// Create the output file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Encode the image to the file in PNG format
	err = png.Encode(file, img)
	if err != nil {
		return err
	}

	return nil
}

func init_test_data(num_images int) string {
	// Get a temporary directory
	base_tmp_dir := os.TempDir()
	dir := base_tmp_dir + "/datago_test"
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		panic(err)
	}

	// Generate random images and save them to the temporary directory
	for i := 0; i < num_images; i++ {
		filename := dir + "/test_image_" + fmt.Sprint(i) + ".png"
		err := generateRandomImage(1024, 1024, filename)
		if err != nil {
			panic(err)
		}
	}

	return dir
}

func TestFilesystemLoad(t *testing.T) {
	// Initialize the test data
	num_images := 10

	test_dir := init_test_data(num_images)
	defer os.RemoveAll(test_dir)

	// Set the environment variable
	os.Setenv("DATAGO_TEST_FILESYSTEM", test_dir)

	// Run the tests
	config := datago.GetDatagoConfig()
	fs_config := datago.GetSourceFileSystemConfig()
	fs_config.RootPath = test_dir
	config.SourceConfig = fs_config

	client := datago.GetClient(config)
	client.Start()

	for i := 0; i < num_images; i++ {
		sample := client.GetSample()
		if sample.ID == "" {
			t.Errorf("GetSample returned an unexpected error")
		}
	}
}
