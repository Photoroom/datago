package datago_test

import (
	"math"
	"math/rand"
	"testing"

	datago "datago/pkg"
)

func TestArAwareTransform(t *testing.T) {
	imageConfig := datago.ImageTransformConfig{
		DefaultImageSize:  1024,
		DownsamplingRatio: 32,
		MinAspectRatio:    0.5,
		MaxAspectRatio:    2.0,
	}

	transform := datago.GetArAwareTransform(imageConfig)

	// For a couple of image sizes, check that we get the expected aspect ratio
	sizes := map[string][2]int{
		"1024x1024": {1024, 1024},
		"704x1440":  {704, 1440},
		"736x1408":  {736, 1408},
		"736x1376":  {736, 1376},
		"768x1344":  {768, 1344},
		"768x1312":  {768, 1312},
		"800x1280":  {800, 1280},
		"832x1248":  {832, 1248},
		"832x1216":  {832, 1216},
		"864x1184":  {864, 1184},
		"896x1152":  {896, 1152},
		"928x1120":  {928, 1120},
		"960x1088":  {960, 1088},
		"992x1056":  {992, 1056},
		"1056x992":  {1056, 992},
		"1088x960":  {1088, 960},
		"1120x928":  {1120, 928},
		"1152x896":  {1152, 896},
		"1184x864":  {1184, 864},
		"1216x832":  {1216, 832},
		"1248x832":  {1248, 832},
		"1280x800":  {1280, 800},
		"1312x768":  {1312, 768},
		"1344x768":  {1344, 768},
		"1376x736":  {1376, 736},
		"1408x736":  {1408, 736},
		"1440x704":  {1440, 704},
	}

	for size, dimensions := range sizes {
		// Fuzz the sizes by a random factor
		fuz := rand.Intn(100) + 1
		fuz2 := rand.Intn(5)

		transformedSize := transform.GetClosestAspectRatio(dimensions[0]*fuz+fuz2, dimensions[1]*fuz)
		if math.Abs(transformedSize-float64(dimensions[0])/float64(dimensions[1])) > 1e-3 {
			t.Error("Aspect ratio mismatch")
			t.Logf("Size: %s, Aspect Ratio: %f", size, transformedSize)
		}
	}
}
