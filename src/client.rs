use crate::generator_files;
use crate::generator_http;
use crate::generator_wds;
use crate::image_processing::ARAwareTransform;
use crate::structs::{DatagoClientConfig, Sample, SourceType};

use crate::structs::DatagoEngine;
use log::{error, info, warn};
use pyo3::prelude::*;

#[pyclass]
pub struct DatagoClient {
    pub is_started: bool,
    source_type: SourceType,
    pub source_config: serde_json::Value,
    pub samples_buffer: usize,
    pub limit: usize,

    // Perf settings
    pub max_connections: usize,
    pub rank: usize,
    pub world_size: usize,

    // Sample processing
    pub image_transform: Option<ARAwareTransform>,
    pub encode_images: bool,
    pub image_to_rgb8: bool, // Convert all images to RGB 8 bits format

    // Holds all the variables related to a running engine
    engine: Option<DatagoEngine>,
}

#[pymethods]
impl DatagoClient {
    #[new]
    pub fn new(str_config: String) -> Self {
        let config: DatagoClientConfig = serde_json::from_str(&str_config).unwrap(); // Ok to panic here, no way we can recover

        if config.rank >= config.world_size {
            panic!("Rank cannot be greater than or equal to world size");
        }

        let mut image_transform: Option<ARAwareTransform> = None;
        let mut encode_images = false;
        let mut image_to_rgb8 = false;
        if let Some(image_config) = config.image_config {
            if image_config.crop_and_resize {
                image_transform = Some(image_config.get_ar_aware_transform());
            }
            encode_images = image_config.pre_encode_images;
            image_to_rgb8 = image_config.image_to_rgb8;
        }

        DatagoClient {
            is_started: false,
            source_type: config.source_type,
            source_config: config.source_config,
            samples_buffer: config.samples_buffer_size,
            limit: config.limit,
            max_connections: 128,
            rank: config.rank,
            world_size: config.world_size,
            image_transform,
            encode_images,
            image_to_rgb8,
            engine: None,
        }
    }

    pub fn start(&mut self) {
        if self.is_started {
            return;
        }

        // In Python, by default the log level is set to "warn", so we do the same here
        // This has no effect, in case the user has previously called initialize_logging().
        initialize_logging(Some("warn".to_string()));

        match self.source_type {
            SourceType::Db => {
                // convert the source_config to a SourceDBConfig
                self.engine = Some(generator_http::orchestrate(self));
            }
            SourceType::File => {
                self.engine = Some(generator_files::orchestrate(self));
            }
            SourceType::WebDataset => {
                self.engine = Some(generator_wds::orchestrate(self));
            }
        }

        self.is_started = true;
    }

    pub fn get_sample(&mut self) -> Option<Sample> {
        if !self.is_started {
            self.start();
        }

        // If no more samples and workers are closed, then wrap it up
        if let Some(engine) = &self.engine {
            if engine.samples_rx.is_closed() {
                info!("No more samples to process, stopping the client");
                self.stop();
                return None;
            }

            // Try to fetch a new sample from the queue
            // The client will timeout if zero sample is received in 5 minutes
            // At this point it will stop and wrap everything up
            const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

            return match engine.samples_rx.recv_timeout(TIMEOUT) {
                Ok(sample) => match sample {
                    Some(sample) => Some(sample),
                    None => {
                        info!("End of stream received, stopping the client");
                        self.stop();
                        None
                    }
                },
                Err(e) => {
                    warn!("Timeout waiting for sample, stopping the client. {}", e);
                    self.stop();
                    None
                }
            };
        }

        None
    }

    pub fn stop(&mut self) {
        if !self.is_started {
            return;
        }

        if let Some(engine) = &mut self.engine {
            let _ = engine.pages_rx.close();
            let _ = engine.samples_tx.close();

            if let Some(pinger) = engine.pinger.take() {
                if pinger.join().is_err() {
                    error!("Failed to join pinger thread");
                }
            }

            if let Some(feeder) = engine.feeder.take() {
                if feeder.join().is_err() {
                    error!("Failed to join feeder thread");
                }
            }

            if let Some(worker) = engine.worker.take() {
                if worker.join().is_err() {
                    error!("Failed to join worker thread");
                }
            }
            self.is_started = false;
        }
    }
}

// Ensure cleanup happens even if stop() wasn't called
impl Drop for DatagoClient {
    fn drop(&mut self) {
        self.stop();
    }
}

#[pyfunction(signature = (log_level=None))]
pub fn initialize_logging(log_level: Option<String>) -> bool {
    // Try to initialize logging, return false if it fails, e.g. if this function is called multiple times.
    if let Some(level) = log_level {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
            .try_init()
            .is_ok()
    } else {
        env_logger::try_init().is_ok()
    }
}

// -------- Unit tests --------

mod tests {
    #[cfg(test)]
    use crate::client::DatagoClient;

    #[cfg(test)]
    use std::collections::HashSet;

    #[cfg(test)]
    use serde_json::json;

    #[cfg(test)]
    use crate::image_processing::ImageTransformConfig;

    #[cfg(test)]
    use crate::structs::ImagePayload;

    #[cfg(test)]
    fn get_test_source() -> String {
        std::env::var("DATAROOM_TEST_SOURCE")
            .expect("DATAROOM_TEST_SOURCE environment variable not set")
    }

    #[cfg(test)]
    fn get_test_config() -> serde_json::Value {
        json!({
            "source_config": {
                "sources": get_test_source(),
                "sources_ne": "",
                "require_images": false,
                "require_embeddings": false,
                "tags": "",
                "tags_ne": "",
                "tags_all": "",
                "tags_ne_all": "",
                "tags_empty": "",
                "has_attributes": "",
                "lacks_attributes": "",
                "has_masks": "",
                "lacks_masks": "",
                "has_latents": "",
                "lacks_latents": "",
                "min_short_edge": 0,
                "max_short_edge": 0,
                "min_pixel_count": -1,
                "max_pixel_count": -1,
                "duplicate_state": -1,
                "attributes": "",
                "random_sampling": false,
                "page_size": 10,
            },
            "limit": 2,
            "rank": 0,
            "world_size": 1,
            "num_threads": 1,
            "max_connections": 1,
            "samples_buffer_size": 1
        })
    }

    #[test]
    fn test_start_stop() {
        let config = get_test_config();
        let mut client = DatagoClient::new(config.to_string());

        client.start();
        client.stop();
    }

    #[test]
    fn test_no_start() {
        let config = get_test_config();
        let mut client = DatagoClient::new(config.to_string());
        client.stop();
    }

    #[test]
    fn test_no_stop() {
        let config = get_test_config();
        let mut client = DatagoClient::new(config.to_string());
        client.start();
    }

    #[test]
    fn test_get_sample() {
        let config = get_test_config();
        let mut client = DatagoClient::new(config.to_string());
        let sample = client.get_sample();

        assert!(sample.is_some());
        assert!(!sample.unwrap().id.is_empty());
    }

    #[test]
    fn test_limit() {
        let limit = 10; // limit < page_size
        let mut config = get_test_config();
        config["limit"] = json!(limit);
        let mut client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());
            assert!(!sample.unwrap().id.is_empty());
        }

        let limit = 100; // limit > page_size
        let mut config = get_test_config();
        config["limit"] = json!(limit);
        let mut client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());
            assert!(!sample.unwrap().id.is_empty());
        }
    }

    #[cfg(test)]
    fn check_image(img: &ImagePayload) {
        assert!(!img.data.is_empty());

        if img.channels > 0 {
            // Raw image
            assert!(img.channels == 3 || img.channels == 1);
            assert!(img.width > 0);
            assert!(img.height > 0);
            assert!(
                img.data.len() * 8
                    == img.width * img.height * img.bit_depth * img.channels as usize
            );
        } else {
            // Encoded image
            assert!(img.width > 0);
            assert!(img.height > 0);
            assert!(!img.data.is_empty());
            assert!(img.channels == -1);

            // Check that we can decode the image
            let _img = image::load_from_memory(&img.data).unwrap();
        }
    }

    #[test]
    fn test_fetch_image() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);
    }

    #[test]
    fn test_extra_fields() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_crop_resize() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        config["image_transform"] = json!(ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 224,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: false,
            image_to_rgb8: false
        });

        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_img_compression() {
        let mut config = get_test_config();
        config["source_config"]["require_images"] = json!(true);
        config["source_config"]["has_latents"] = "masked_image".into();
        config["source_config"]["has_masks"] = "segmentation_mask".into();

        config["image_transform"] = json!(ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 224,
            downsampling_ratio: 16,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: true, // new part being tested
            image_to_rgb8: false
        });

        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        check_image(&sample.image);

        assert!(sample.additional_images.contains_key("masked_image"));
        check_image(&sample.additional_images["masked_image"]);

        assert!(sample.masks.contains_key("segmentation_mask"));
        check_image(&sample.masks["segmentation_mask"]);
    }

    #[test]
    fn test_tags() {
        let mut config = get_test_config();
        let tag = "v4_trainset_hq";

        // Test positive tags
        config["source_config"]["tags"] = tag.into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(sample.tags.contains(&tag.to_string()));
        client.stop();

        // Test negative tags
        config["source_config"]["tags"] = "".into();
        config["source_config"]["tags_ne"] = "v4_trainset_hq".into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(!sample.tags.contains(&tag.to_string()));
        client.stop();
    }

    #[test]
    fn test_tags_all() {
        let mut config = get_test_config();
        let tags = "v4_trainset_hq,photo";
        config["source_config"]["tags_all"] = tags.into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Check that sample.tags contains all the tags in the tags string
        for tag in tags.split(',') {
            assert!(sample.tags.contains(&tag.to_string()));
        }
        client.stop();
    }

    #[test]
    fn test_tags_ne() {
        let mut config = get_test_config();
        let tags = "v4_trainset_hq,photo";
        config["source_config"]["tags_ne"] = tags.into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Check that sample.tags does not contain any of the tags in the tags string
        println!("{:?}", sample.tags);
        for tag in tags.split(',') {
            assert!(!sample.tags.contains(&tag.to_string()));
        }
        client.stop();
    }

    #[test]
    fn test_tags_empty() {
        let mut config = get_test_config();
        config["source_config"]["tags_empty"] = "true".into();
        config["source_config"]["tags_ne"] = "".into();
        config["source_config"]["tags"] = "".into();

        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some(), "Sample should be present");

        let sample = sample.unwrap();
        assert!(sample.tags.is_empty(), "Tags should be empty");
        client.stop();
    }

    #[test]
    fn test_tags_ne_all() {
        let mut config = get_test_config();
        let tag1 = "photo";
        let tag2 = "graphic";
        config["source_config"]["tags_ne_all"] = format!("{},{}", tag1, tag2).into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        // Assert that the sample does not contain both tags at the same time
        let has_first = sample.tags.contains(&tag1.to_string());
        let has_second = sample.tags.contains(&tag2.to_string());
        assert!(
            !(has_first && has_second),
            "Sample should not contain both tags at the same time"
        );
        client.stop();
    }

    #[test]
    fn test_attributes_filter() {
        let mut config = get_test_config();
        config["source_config"]["attributes"] = "aesthetic_score__gte:0.5".into();
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(sample.attributes.contains_key("aesthetic_score"));
        assert!(sample.attributes["aesthetic_score"].as_f64().unwrap() >= 0.5);
        client.stop();
    }

    #[test]
    fn test_pixel_count_filter() {
        let mut config = get_test_config();
        config["source_config"]["min_pixel_count"] = 1000000.into();
        config["source_config"]["max_pixel_count"] = 2000000.into();
        config["source_config"]["require_images"] = json!(true);
        let mut client = DatagoClient::new(config.to_string());

        let sample = client.get_sample();
        assert!(sample.is_some());

        let sample = sample.unwrap();
        assert!(!sample.id.is_empty());
        assert!(sample.image.width * sample.image.height >= 1000000);
        assert!(sample.image.width * sample.image.height <= 2000000);
        client.stop();
    }

    #[test]
    fn test_multiple_sources() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["sources"] = "LAION_ART,LAION_AESTHETICS".into();
        config["limit"] = json!(limit);

        let mut client = DatagoClient::new(config.to_string());
        let sources = config["source_config"]["sources"]
            .as_str()
            .unwrap()
            .split(",")
            .collect::<Vec<&str>>();

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());

            let sample = sample.unwrap();
            assert!(!sample.id.is_empty());
            println!("{}", sample.source);
            assert!(sources.contains(&sample.source.as_str()));
        }
    }

    #[test]
    fn test_sources_ne() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["sources"] = "LAION_ART,LAION_AESTHETICS".into();
        config["source_config"]["sources_ne"] = "LAION_ART".into();
        config["limit"] = json!(limit);

        println!("{}", config);
        let mut client = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            let sample = client.get_sample();
            assert!(sample.is_some());

            let sample = sample.unwrap();
            assert!(!sample.id.is_empty());
            assert!(sample.source == "LAION_AESTHETICS");
        }
    }

    #[test]
    fn test_random_sampling() {
        let limit = 10;
        let mut config = get_test_config();
        config["source_config"]["random_sampling"] = json!(true);
        config["limit"] = json!(limit);

        // Fill in two sets with some results, and check that they are different
        let mut sample_set_1: HashSet<String> = HashSet::new();
        let mut sample_set_2: HashSet<String> = HashSet::new();

        let mut client_1 = DatagoClient::new(config.to_string());
        let mut client_2 = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            sample_set_1.insert(client_1.get_sample().unwrap().id);
        }

        for _ in 0..limit {
            sample_set_2.insert(client_2.get_sample().unwrap().id);
        }
        assert!(sample_set_1 != sample_set_2);
    }

    #[test]
    #[should_panic(expected = "Rank cannot be greater than or equal to world size")]
    fn test_broken_ranks() {
        let mut config = get_test_config();
        config["world_size"] = json!(2);

        // Check that we assert if rank >= world_size
        config["rank"] = json!(2);

        let mut client = DatagoClient::new(config.to_string());
        client.start();
        client.stop();
    }

    #[test]
    fn test_ranks() {
        let mut config = get_test_config();
        let limit = 100;
        config["source_config"]["require_images"] = json!(false);
        config["world_size"] = json!(2);
        config["limit"] = json!(limit);

        // Fill in two sets with some results, and check that they are completely different
        let mut sample_set_1: HashSet<String> = HashSet::new();
        let mut sample_set_2: HashSet<String> = HashSet::new();

        config["rank"] = json!(0);
        let mut client_1 = DatagoClient::new(config.to_string());

        config["rank"] = json!(1);
        let mut client_2 = DatagoClient::new(config.to_string());

        for _ in 0..limit {
            sample_set_1.insert(client_1.get_sample().unwrap().id);
        }

        for _ in 0..limit {
            sample_set_2.insert(client_2.get_sample().unwrap().id);
        }

        // Check that the two sets are completely different
        assert!(sample_set_1.intersection(&sample_set_2).count() == 0);
    }
}
