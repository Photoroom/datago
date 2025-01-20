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
	PreEncodeImages   bool
}

func buildImageSizeList(defaultImageSize int, downsamplingRatio int, minAspectRatio float64, maxAspectRatio float64) []ImageSize {
	patchSize := defaultImageSize / downsamplingRatio
	patchSizeSq := float64(patchSize * patchSize)
	var imgSizes []ImageSize

	minPatchW := int(math.Ceil(patchSizeSq * minAspectRatio))
	maxPatchW := int(math.Floor(patchSizeSq * maxAspectRatio))

	for patchW := minPatchW; patchW <= maxPatchW; patchW++ { // go over all possible downsampled image widths
		patchH := int(math.Floor(patchSizeSq / float64(patchW))) // get max height
		imgW, imgH := patchW*downsamplingRatio, patchH*downsamplingRatio
		imgSizes = append(imgSizes, ImageSize{imgW, imgH})
	}

	minPatchH := int(math.Ceil(math.Sqrt(patchSizeSq * 1.0 / maxAspectRatio)))
	maxPatchH := int(math.Floor(math.Sqrt(patchSizeSq * 1.0 / minAspectRatio)))
	for patchH := minPatchH; patchH <= maxPatchH; patchH++ { // go over all possible downsampled image heights
		patchW := int(math.Floor(patchSizeSq / float64(patchH))) // get max width
		imgW, imgH := patchW*downsamplingRatio, patchH*downsamplingRatio
		imgSizes = append(imgSizes, ImageSize{imgW, imgH})
	}

	return imgSizes
}

func newARAwareTransform(imageConfig ImageTransformConfig) *ARAwareTransform {
	// Build the image size list
	imgSizes := buildImageSizeList(imageConfig.DefaultImageSize, imageConfig.DownsamplingRatio, imageConfig.MinAspectRatio, imageConfig.MaxAspectRatio)

	// Fill in the map table to match aspect ratios and image sizes
	aspectRatioToSize := make(map[float64]ImageSize)
	for _, size := range imgSizes {
		aspectRatioToSize[size.AspectRatio()] = size
	}

	//
	return &ARAwareTransform{
		defaultImageSize:  imageConfig.DefaultImageSize,
		downsamplingRatio: imageConfig.DownsamplingRatio,
		minAspectRatio:    imageConfig.MinAspectRatio,
		maxAspectRatio:    imageConfig.MaxAspectRatio,
		targetImageSizes:  imgSizes,
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
