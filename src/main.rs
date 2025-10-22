use clap::{Arg, Command};
use log::{error, info};
use prettytable::{row, Table};
use serde_json::json;
use std::collections::HashMap;
mod client;
mod generator_files;
mod generator_http;
mod generator_wds;

mod image_processing;
mod structs;
mod worker_files;
mod worker_http;
mod worker_wds;

fn main() {
    client::initialize_logging(Some("info".to_string()));

    // -----------------------------------------------------------------
    // Handle CLI arguments
    let matches = Command::new("Datago-rs")
        .version("1.0")
        .author("ML Team <ml@photoroom.com>")
        .about("Processes data with multiple threads")
        .arg(
            Arg::new("backlog")
                .short('b')
                .long("backlog")
                .help("Sets the samples_buffer_size size")
                .default_value("128"),
        )
        .arg(
            Arg::new("num_samples")
                .short('n')
                .long("samples")
                .help("Sets the number of samples")
                .default_value("3000"),
        )
        .arg(
            Arg::new("page_size")
                .short('p')
                .long("page_size")
                .help("Sets the page size")
                .default_value("500"),
        )
        .arg(
            Arg::new("sources")
                .short('s')
                .long("sources")
                .help("Sets the sources used to query the DB with")
                .default_value("COYO"),
        )
        .arg(
            Arg::new("crop_and_resize")
                .short('c')
                .long("crop_and_resize")
                .help("Align all the image sizes, crop and resize as required")
                .required(false),
        )
        .arg(
            Arg::new("save_samples")
                .long("save_samples")
                .help("Save the samples to disk")
                .required(false),
        )
        .get_matches();

    let sources = matches
        .get_one::<String>("sources")
        .unwrap()
        .parse::<String>()
        .unwrap();
    let samples_buffer_size = matches
        .get_one::<String>("backlog")
        .unwrap()
        .parse::<usize>()
        .unwrap_or(256);
    let page_size = matches
        .get_one::<String>("page_size")
        .unwrap()
        .parse::<usize>()
        .unwrap_or(500);
    let num_samples = matches
        .get_one::<String>("num_samples")
        .unwrap()
        .parse::<usize>()
        .unwrap_or(10_000);

    let crop_and_resize = matches.contains_id("crop_and_resize");
    let save_samples = matches.contains_id("save_samples");

    // -----------------------------------------------------------------
    // Get the appropriate image transformation
    let image_transform_config = if crop_and_resize {
        Some(image_processing::ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 1024,
            downsampling_ratio: 32,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: true,
            image_to_rgb8: false,
            encode_format: image_processing::EncodeFormat::default(),
            jpeg_quality: 92,
        })
    } else {
        None
    };

    let config = json!({
        "source_config": {
            "sources": sources,
            "require_images": true,
            "page_size": page_size,
        },
        "image_config": image_transform_config,
        "limit": num_samples,
        "rank": 0,
        "world_size": 1,
        "samples_buffer_size": samples_buffer_size
    });

    info!("{config}");

    let mut client = client::DatagoClient::new(config.to_string());

    // -----------------------------------------------------------------
    let mut size_buckets: HashMap<String, i32> = HashMap::new();
    let start_time = std::time::Instant::now();
    let mut rolling_time = std::time::Instant::now();

    let mut num_samples_received = 0;

    client.start();
    for i in 0..num_samples {
        if let Some(sample) = client.get_sample() {
            if crop_and_resize {
                // Count the number of samples per size
                let size = format!("{}x{}", sample.image.width, sample.image.height);
                let count = size_buckets.entry(size).or_insert(0);
                *count += 1;
            }
            if save_samples {
                let img = image::load_from_memory(&sample.image.data).unwrap();
                let filename = format!("sample_{num_samples_received:?}.jpg");
                img.save(filename).unwrap();
            }
            num_samples_received += 1;
        } else {
            error!("Failed to get a sample.");
            error!("Make sure you use a non-empty source and you set the DATAROOM_API_URL and DATAROOM_API_KEY env vars correctly");
            break;
        }

        if i % 100 == 0 {
            info!(
                "{i}: Samples/s {:.2}",
                100_f64 / rolling_time.elapsed().as_secs_f64()
            );
            rolling_time = std::time::Instant::now();
        }
    }
    client.stop();
    info!("All samples processed. Got {num_samples_received:?} samples\n");

    // Report the per-bucket occupancy, good sanity check
    if crop_and_resize {
        let mut size_buckets_str = String::from("Size buckets:\n");
        for (size, count) in size_buckets.iter() {
            size_buckets_str.push_str(&format!("{size}: {count}\n"));
        }
        info!("{size_buckets_str}");
    }

    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let fps = num_samples as f64 / elapsed_secs;

    let mut table = Table::new();
    table.add_row(row!["Total Samples", num_samples]);
    table.add_row(row!["Execution time", format!("{:.2}", elapsed_secs)]);
    table.add_row(row!["Samples / s", format!("{:.2}", fps)]);
    table.add_row(row!["Threads", num_cpus::get()]);
    table.printstd();
}
