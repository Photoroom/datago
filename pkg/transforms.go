package datago

import (
	"fmt"
	"math"

	"github.com/davidbyttow/govips/v2/vips"
)

type ImageSize struct {
	// Making it explicit how we store width and height, guarding against potential confusion
	Width  int
	Height int
}

func (s *ImageSize) AspectRatio() float64 {
	// Specifying how we compute the aspect ratio explicitly, since height/width and width/height are both valid options
	return float64(s.Width) / float64(s.Height)
}

type ARAwareTransform struct {
	defaultImageSize  int
	downsamplingRatio int
	minAspectRatio    float64
	maxAspectRatio    float64
	targetImageSizes  []ImageSize // list of [width, height] pairs
	aspectRatioToSize map[float64]ImageSize
}

func buildImageSizeList(defaultImageSize int, downsamplingRatio int, minAspectRatio float64, maxAspectRatio float64) []ImageSize {
	patch_size := defaultImageSize / downsamplingRatio
	patch_size_sq := float64(patch_size * patch_size)
	var image_list []ImageSize

	min_patch_w := int(math.Ceil(patch_size_sq * minAspectRatio))
	max_patch_w := int(math.Floor(patch_size_sq * maxAspectRatio))

	for patch_w := min_patch_w; patch_w <= max_patch_w; patch_w++ { // go over all possible downsampled image widths
		patch_h := int(math.Floor(patch_size_sq / float64(patch_w))) // get max height
		img_w, img_h := patch_w*downsamplingRatio, patch_h*downsamplingRatio
		image_list = append(image_list, ImageSize{img_w, img_h})
	}

	min_patch_h := int(math.Ceil(math.Sqrt(patch_size_sq * 1.0 / maxAspectRatio)))
	max_patch_h := int(math.Floor(math.Sqrt(patch_size_sq * 1.0 / minAspectRatio)))
	for patch_h := min_patch_h; patch_h <= max_patch_h; patch_h++ { // go over all possible downsampled image heights
		patch_w := int(math.Floor(patch_size_sq / float64(patch_h))) // get max width
		img_w, img_h := patch_w*downsamplingRatio, patch_h*downsamplingRatio
		image_list = append(image_list, ImageSize{img_w, img_h})
	}

	return image_list
}

func newARAwareTransform(imageConfig ImageTransformConfig) *ARAwareTransform {
	// Build the image size list
	image_list := buildImageSizeList(imageConfig.DefaultImageSize, imageConfig.DownsamplingRatio, imageConfig.MinAspectRatio, imageConfig.MaxAspectRatio)

	// Fill in the map table to match aspect ratios and image sizes
	aspectRatioToSize := make(map[float64]ImageSize)
	for _, size := range image_list {
		aspectRatioToSize[size.AspectRatio()] = size
	}

	//
	return &ARAwareTransform{
		defaultImageSize:  imageConfig.DefaultImageSize,
		downsamplingRatio: imageConfig.DownsamplingRatio,
		minAspectRatio:    imageConfig.MinAspectRatio,
		maxAspectRatio:    imageConfig.MaxAspectRatio,
		targetImageSizes:  image_list,
		aspectRatioToSize: aspectRatioToSize,
	}
}

func (t *ARAwareTransform) getClosestAspectRatio(imageWidth int, imageHeight int) float64 {
	// Find the closest aspect ratio to the given aspect ratio
	if len(t.aspectRatioToSize) == 0 {
		fmt.Println("Aspect ratio to size map is empty")
		panic("Aspect ratio to size map is empty")
	}

	image_size := ImageSize{Width: imageWidth, Height: imageHeight}
	aspectRatio := image_size.AspectRatio()

	// No choice but walking through all the possible values. Maybe possible to optimize this. in Go
	minDiff := math.MaxFloat64
	closestAspectRatio := 0.0
	for ar := range t.aspectRatioToSize {
		diff := math.Abs(ar - aspectRatio)
		if diff < minDiff {
			minDiff = diff
			closestAspectRatio = ar
		}
	}

	return closestAspectRatio
}

func (t *ARAwareTransform) cropAndResizeToClosestAspectRatio(image *vips.ImageRef, referenceAR float64) (float64, error) {

	// Get the closest aspect ratio
	if referenceAR <= 0. {
		referenceAR = t.getClosestAspectRatio(image.Width(), image.Height())
	}

	// Desired target size is a lookup away, this is pre-computed/bucketed
	targetSize := t.aspectRatioToSize[referenceAR]

	// Trust libvips to do resize and crop in one go. Note that jpg decoding happens here and can fail
	err := image.ThumbnailWithSize(targetSize.Width, targetSize.Height, vips.InterestingCentre, vips.SizeBoth)
	return referenceAR, err
}
