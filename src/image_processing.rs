use crate::structs::ImagePayload;
use fast_image_resize as fr;
use fast_image_resize::images::Image;
use fast_image_resize::{IntoImageView, ResizeOptions, Resizer};
use image::ImageEncoder;
use log::debug;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Cursor;
// --- Sample data structures - these will be exposed to the Python world ---------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageTransformConfig {
    pub crop_and_resize: bool,
    pub default_image_size: u32,
    pub downsampling_ratio: u32,
    pub min_aspect_ratio: f64,
    pub max_aspect_ratio: f64,

    #[serde(default)]
    pub pre_encode_images: bool,

    #[serde(default)]
    pub image_to_rgb8: bool, // Convert all images to RGB 8 bits format
}

impl ImageTransformConfig {
    pub fn get_ar_aware_transform(&self) -> ARAwareTransform {
        let target_image_sizes = build_image_size_list(
            self.default_image_size,
            self.downsampling_ratio,
            self.min_aspect_ratio,
            self.max_aspect_ratio,
        );

        debug!(
            "Cropping and resizing images. Target image sizes:\n{:?}\n",
            target_image_sizes
        );

        let mut aspect_ratio_to_size = std::collections::HashMap::new();
        for img_size in &target_image_sizes {
            let aspect_ratio = format!("{:.3}", img_size.0 as f64 / img_size.1 as f64);
            aspect_ratio_to_size.insert(aspect_ratio, *img_size);
        }

        let mut aspect_ratios: Vec<(f64, String)> = aspect_ratio_to_size
            .keys()
            .filter_map(|k| k.parse::<f64>().ok().map(|f| (f, k.clone())))
            .collect();
        aspect_ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        ARAwareTransform {
            aspect_ratio_to_size,
            aspect_ratios,
        }
    }
}

#[derive(Clone)]
pub struct ARAwareTransform {
    aspect_ratio_to_size: HashMap<String, (u32, u32)>,
    // Cache for fast aspect ratio lookups
    aspect_ratios: Vec<(f64, String)>,
}

pub fn aspect_ratio_to_str(size: (u32, u32)) -> String {
    let ar_str = format!("{:.3}", size.0 as f64 / size.1 as f64);
    ar_str
}

fn image_to_dyn_image(dst_image: &Image) -> image::DynamicImage {
    // Convert the fast_image_resize::Image back to image::DynamicImage
    let width = dst_image.width();
    let height = dst_image.height();
    let pixels = dst_image.buffer().to_vec();

    match dst_image.pixel_type() {
        fr::PixelType::U8x3 => {
            let img_buffer = image::RgbImage::from_raw(width, height, pixels).unwrap();
            image::DynamicImage::ImageRgb8(img_buffer)
        }
        fr::PixelType::U8x4 => {
            let img_buffer = image::RgbaImage::from_raw(width, height, pixels).unwrap();
            image::DynamicImage::ImageRgba8(img_buffer)
        }

        fr::PixelType::U8 | fr::PixelType::U8x2 => {
            let img_buffer = image::GrayImage::from_raw(width, height, pixels).unwrap();
            image::DynamicImage::ImageLuma8(img_buffer)
        }

        _ => panic!("Unsupported pixel type: {:?}", dst_image.pixel_type()),
    }
}

fn build_image_size_list(
    default_image_size: u32,
    downsampling_ratio: u32,
    min_aspect_ratio: f64,
    max_aspect_ratio: f64,
) -> Vec<(u32, u32)> {
    let patch_size = default_image_size / downsampling_ratio;
    let patch_size_sq = (patch_size * patch_size) as f64;
    let mut img_sizes: Vec<(u32, u32)> = Vec::new();

    let min_patch_w = (patch_size_sq * min_aspect_ratio).sqrt().ceil() as u32;
    let max_patch_w = (patch_size_sq * max_aspect_ratio).sqrt().floor() as u32;

    for patch_w in min_patch_w..=max_patch_w {
        let patch_h = (patch_size_sq / patch_w as f64).floor() as u32;
        let img_w = patch_w * downsampling_ratio;
        let img_h = patch_h * downsampling_ratio;
        img_sizes.push((img_w, img_h));
    }

    let min_patch_h = (patch_size_sq / max_aspect_ratio).sqrt().ceil() as u32;
    let max_patch_h = (patch_size_sq / min_aspect_ratio).sqrt().floor() as u32;

    for patch_h in min_patch_h..=max_patch_h {
        let patch_w = (patch_size_sq / patch_h as f64).floor() as u32;
        let img_w = patch_w * downsampling_ratio;
        let img_h = patch_h * downsampling_ratio;
        img_sizes.push((img_w, img_h));
    }

    img_sizes
}

impl ARAwareTransform {
    pub fn get_closest_aspect_ratio(&self, image_width: i32, image_height: i32) -> String {
        if self.aspect_ratios.is_empty() {
            panic!("Aspect ratio to size map is empty");
        }

        let target_ar = image_width as f64 / image_height as f64;

        // Binary search for closest aspect ratio
        match self
            .aspect_ratios
            .binary_search_by(|&(ar, _)| ar.partial_cmp(&target_ar).unwrap())
        {
            Ok(idx) => self.aspect_ratios[idx].1.clone(),
            Err(idx) => {
                if idx == 0 {
                    self.aspect_ratios[0].1.clone()
                } else if idx == self.aspect_ratios.len() {
                    self.aspect_ratios[self.aspect_ratios.len() - 1].1.clone()
                } else {
                    // Choose the closer of the two adjacent ratios
                    let left_diff = (target_ar - self.aspect_ratios[idx - 1].0).abs();
                    let right_diff = (self.aspect_ratios[idx].0 - target_ar).abs();
                    if left_diff < right_diff {
                        self.aspect_ratios[idx - 1].1.clone()
                    } else {
                        self.aspect_ratios[idx].1.clone()
                    }
                }
            }
        }
    }

    pub async fn crop_and_resize(
        &self,
        image: &image::DynamicImage,
        aspect_ratio_input: Option<&String>,
    ) -> image::DynamicImage {
        let aspect_ratio = match aspect_ratio_input {
            Some(ar) => ar.to_string(),
            None => self.get_closest_aspect_ratio(image.width() as i32, image.height() as i32),
        };

        if let Some(target_size) = self.aspect_ratio_to_size.get(&aspect_ratio) {
            // Check if resize is actually needed
            if image.width() == target_size.0 && image.height() == target_size.1 {
                image.clone()
            } else {
                let image_pixel_type = image.pixel_type().unwrap();
                if image_pixel_type == fr::PixelType::U8
                    || image_pixel_type == fr::PixelType::U8x2
                    || image_pixel_type == fr::PixelType::U8x3
                    || image_pixel_type == fr::PixelType::U8x4
                {
                    // Fast path with the fast_image_resize library

                    // Calculate the scale factors for both dimensions
                    let scale_x = target_size.0 as f64 / image.width() as f64;
                    let scale_y = target_size.1 as f64 / image.height() as f64;

                    // Use the larger scale factor to ensure one dimension matches target
                    // and the other is at least the target size
                    let scale = scale_x.max(scale_y);

                    let new_width = (image.width() as f64 * scale).round() as u32;
                    let new_height = (image.height() as f64 * scale).round() as u32;

                    let mut dst_image =
                        Image::new(new_width, new_height, image.pixel_type().unwrap());

                    let mut resizer = Resizer::new();

                    let resize_options = ResizeOptions::new()
                        .resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3));

                    resizer
                        .resize(image, &mut dst_image, &resize_options)
                        .unwrap();

                    // Crop the resized image to the target size
                    let mut final_image =
                        Image::new(target_size.0, target_size.1, image.pixel_type().unwrap());

                    let crop_box = fr::CropBox::fit_src_into_dst_size(
                        new_width,
                        new_height,
                        target_size.0,
                        target_size.1,
                        None, // Default to center crop
                    );

                    resizer
                        .resize(
                            &dst_image,
                            &mut final_image,
                            &ResizeOptions::new().crop(
                                crop_box.left,
                                crop_box.top,
                                crop_box.width,
                                crop_box.height,
                            ),
                        )
                        .unwrap();

                    image_to_dyn_image(&final_image)
                } else {
                    image.resize_to_fill(
                        target_size.0,
                        target_size.1,
                        image::imageops::FilterType::Lanczos3,
                    )
                }
            }
        } else {
            panic!("Aspect ratio not found in aspect ratio to size map");
        }
    }
}

// ------------------------------------------------------------------------
pub async fn image_to_payload(
    mut image: image::DynamicImage,
    img_tfm: &Option<ARAwareTransform>,
    aspect_ratio: &String,
    encode_images: bool,
    img_to_rgb8: bool,
) -> Result<ImagePayload, image::ImageError> {
    let original_height = image.height() as usize;
    let original_width = image.width() as usize;
    let mut channels = image.color().channel_count() as i8;
    let mut bit_depth =
        (image.color().bits_per_pixel() / image.color().channel_count() as u16) as usize;

    // Optionally transform the additional image in the same way the main image was
    if let Some(img_tfm) = img_tfm {
        let aspect_ratio_input = if aspect_ratio.is_empty() {
            None
        } else {
            Some(aspect_ratio)
        };
        image = img_tfm.crop_and_resize(&image, aspect_ratio_input).await;
    }

    let height = image.height() as usize;
    let width = image.width() as usize;

    // Image to RGB8 if requested
    if img_to_rgb8 && image.color() != image::ColorType::Rgb8 {
        image = image::DynamicImage::ImageRgb8(image.to_rgb8());
        bit_depth = 8;
        channels = 3;
        assert!((image.color().bits_per_pixel() / image.color().channel_count() as u16) == 8);
    }

    // Encode the image if needed
    let mut image_bytes: Vec<u8> = Vec::new();
    if encode_images {
        // Pre-allocate buffer based on image size estimate
        image_bytes.reserve(width * height * channels as usize);

        // Use the encoder directly with the raw bytes
        image::codecs::png::PngEncoder::new_with_quality(
            &mut Cursor::new(&mut image_bytes),
            image::codecs::png::CompressionType::Fast,
            image::codecs::png::FilterType::Adaptive,
        )
        .write_image(
            image.as_bytes(),
            image.width(),
            image.height(),
            image.color().into(),
        )
        .map_err(|e| {
            image::ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        channels = -1; // Signal the fact that the image is encoded
    } else {
        image_bytes = image.into_bytes();
    }

    Ok(ImagePayload {
        data: image_bytes,
        original_height,
        original_width,
        height,
        width,
        channels,
        bit_depth,
    })
}

// ------------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use image::GenericImageView;

    #[tokio::test]
    async fn test_aspect_ratio_transform() {
        let config = ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 224,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: false,
            image_to_rgb8: false,
        };

        let transform = config.get_ar_aware_transform();

        // Test getting closest aspect ratio
        assert_eq!(transform.get_closest_aspect_ratio(100, 100), "1.000");
        assert_eq!(transform.get_closest_aspect_ratio(200, 100), "1.900");
        assert_eq!(transform.get_closest_aspect_ratio(100, 200), "0.526");

        // Test image resizing
        let img = DynamicImage::new_rgb8(300, 200);
        let resized = transform
            .crop_and_resize(&img, Some(&"1.000".to_string()))
            .await;
        assert_eq!(resized.dimensions(), (224, 224));

        let resized = transform
            .crop_and_resize(&img, Some(&"1.900".to_string()))
            .await;
        assert_eq!(resized.dimensions(), (304, 160));

        // Test empty aspect ratio input (should use closest)
        let img = DynamicImage::new_rgb8(400, 200);
        let resized = transform.crop_and_resize(&img, None).await;
        assert_eq!(resized.dimensions(), (304, 160));
    }

    #[test]
    fn test_build_image_size_list() {
        let sizes = build_image_size_list(224, 16, 0.5, 2.0);
        assert!(!sizes.is_empty());

        // Check all sizes respect min/max aspect ratio
        for (w, h) in sizes {
            let ar = w as f64 / h as f64;
            assert!((0.5..=2.0).contains(&ar));

            // Check dimensions are multiples of downsampling ratio
            assert_eq!(w % 16, 0);
            assert_eq!(h % 16, 0);
        }
    }
}
