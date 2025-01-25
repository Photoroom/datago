// --- Sample data structures - these will be exposed to the Python world ---------------------------------------------------------------------------------------------------------------------------------------------------------------
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageTransformConfig {
    pub crop_and_resize: bool,
    pub default_image_size: i32,
    pub downsampling_ratio: i32,
    pub min_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    pub pre_encode_images: bool,
}

impl ImageTransformConfig {
    pub fn get_ar_aware_transform(&self) -> ARAwareTransform {
        let target_image_sizes = build_image_size_list(
            self.default_image_size,
            self.downsampling_ratio,
            self.min_aspect_ratio,
            self.max_aspect_ratio,
        );

        println!("Target image sizes: {:?}\n", target_image_sizes);

        let mut aspect_ratio_to_size = std::collections::HashMap::new();
        for img_size in &target_image_sizes {
            let aspect_ratio = format!("{:.3}", img_size.0 as f64 / img_size.1 as f64);
            aspect_ratio_to_size.insert(aspect_ratio, *img_size);
        }

        ARAwareTransform {
            aspect_ratio_to_size,
        }
    }
}

#[derive(Clone)]
pub struct ARAwareTransform {
    aspect_ratio_to_size: std::collections::HashMap<String, (i32, i32)>, // list of [width, height] pairs
}

pub fn aspect_ratio_to_str(size: (i32, i32)) -> String {
    let ar_str = format!("{:.3}", size.0 as f64 / size.1 as f64);
    ar_str
}

fn build_image_size_list(
    default_image_size: i32,
    downsampling_ratio: i32,
    min_aspect_ratio: f64,
    max_aspect_ratio: f64,
) -> Vec<(i32, i32)> {
    let patch_size = default_image_size / downsampling_ratio;
    let patch_size_sq = (patch_size * patch_size) as f64;
    let mut img_sizes: Vec<(i32, i32)> = Vec::new();

    let min_patch_w = (patch_size_sq * min_aspect_ratio).sqrt().ceil() as i32;
    let max_patch_w = (patch_size_sq * max_aspect_ratio).sqrt().floor() as i32;

    for patch_w in min_patch_w..=max_patch_w {
        let patch_h = (patch_size_sq / patch_w as f64).floor() as i32;
        let img_w = patch_w * downsampling_ratio;
        let img_h = patch_h * downsampling_ratio;
        img_sizes.push((img_w, img_h));
    }

    let min_patch_h = (patch_size_sq / max_aspect_ratio).sqrt().ceil() as i32;
    let max_patch_h = (patch_size_sq / min_aspect_ratio).sqrt().floor() as i32;

    for patch_h in min_patch_h..=max_patch_h {
        let patch_w = (patch_size_sq / patch_h as f64).floor() as i32;
        let img_w = patch_w * downsampling_ratio;
        let img_h = patch_h * downsampling_ratio;
        img_sizes.push((img_w, img_h));
    }

    img_sizes
}

impl ARAwareTransform {
    pub fn get_closest_aspect_ratio(&self, image_width: i32, image_height: i32) -> String {
        if self.aspect_ratio_to_size.is_empty() {
            panic!("Aspect ratio to size map is empty");
        }

        let aspect_ratio_float = aspect_ratio_to_str((image_width, image_height))
            .parse::<f64>()
            .unwrap();

        let mut min_diff = f64::MAX;
        let mut closest_aspect_ratio = String::new();
        for ar_key in self.aspect_ratio_to_size.keys() {
            let ar_key_float: f64 = ar_key.parse().unwrap();
            let diff = (ar_key_float - aspect_ratio_float).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_aspect_ratio = ar_key.clone();
            }
        }

        closest_aspect_ratio.to_string()
    }

    pub fn crop_and_resize(
        &self,
        image: &image::DynamicImage,
        aspect_ratio_input: &String,
    ) -> image::DynamicImage {
        let aspect_ratio = if aspect_ratio_input.is_empty() {
            self.get_closest_aspect_ratio(image.width() as i32, image.height() as i32)
        } else {
            aspect_ratio_input.to_string()
        };

        let target_size = self.aspect_ratio_to_size.get(&aspect_ratio).unwrap();
        let image_crop_and_resize = image.resize_to_fill(
            target_size.0 as u32,
            target_size.1 as u32,
            image::imageops::FilterType::Lanczos3,
        );

        image_crop_and_resize
    }
}
