use datago::DatagoClient;
use datago::ImagePayload;
use datago::ImageTransformConfig;

use serde_json::json;
use std::collections::HashSet;

fn get_test_source() -> String {
    std::env::var("DATAROOM_TEST_SOURCE")
        .expect("DATAROOM_TEST_SOURCE environment variable not set")
}

fn get_test_config() -> serde_json::Value {
    json!({
        "source_config": {
            "sources": get_test_source(),
            "sources_ne": "",
            "require_images": false,
            "require_embeddings": false,
            "tags": "",
            "tags_ne": "",
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
    assert!(sample.unwrap().id != "");
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
        assert!(sample.unwrap().id != "");
    }

    let limit = 100; // limit > page_size
    let mut config = get_test_config();
    config["limit"] = json!(limit);
    let mut client = DatagoClient::new(config.to_string());

    for _ in 0..limit {
        let sample = client.get_sample();
        assert!(sample.is_some());
        assert!(sample.unwrap().id != "");
    }
}

fn check_image(img: &ImagePayload) {
    assert!(img.data.len() > 0);

    if img.channels > 0 {
        // Raw image
        assert!(img.channels == 3 || img.channels == 1);
        assert!(img.width > 0);
        assert!(img.height > 0);
        assert!(img.data.len() * 8 == img.width * img.height * img.bit_depth as usize);
    } else {
        // Encoded image
        assert!(img.width > 0);
        assert!(img.height > 0);
        assert!(img.data.len() > 0);
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
    assert!(sample.id.len() > 0);
    assert!(sample.tags.contains(&tag.to_string()));
    client.stop();

    // Test negative tags
    config["source_config"]["tags"] = "".into();
    config["source_config"]["tags_ne"] = "v4_trainset_hq".into();
    let mut client = DatagoClient::new(config.to_string());

    let sample = client.get_sample();
    assert!(sample.is_some());

    let sample = sample.unwrap();
    assert!(sample.id.len() > 0);
    assert!(!sample.tags.contains(&tag.to_string()));
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
        assert!(sample.id.len() > 0);
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
        assert!(sample.id.len() > 0);
        println!("{}", sample.source);
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
