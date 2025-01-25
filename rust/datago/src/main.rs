use clap::{Arg, Command};
use prettytable::{row, Table};

mod client;
mod generator_http;
mod image_processing;
mod worker_http;

fn main() {
    // -----------------------------------------------------------------
    // Handle CLI arguments
    let matches = Command::new("Datago-rs")
        .version("1.0")
        .author("Author Name <ml@photoroom.com>")
        .about("Processes data with multiple threads")
        .arg(
            Arg::new("num_threads")
                .short('t')
                .long("threads")
                .help("Sets the number of threads")
                .default_value("32"),
        )
        .arg(
            Arg::new("backlog")
                .short('b')
                .long("backlog")
                .help("Sets the backlog size")
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
            Arg::new("source")
                .short('s')
                .long("source")
                .help("Sets the source used to query the DB with")
                .default_value("COYO"),
        )
        .arg(
            Arg::new("crop_and_resize")
                .short('c')
                .long("crop_and_resize")
                .help("Align all the image sizes, crop and resize as required")
                .required(false),
        )
        .get_matches();

    let source = matches
        .get_one::<String>("source")
        .unwrap()
        .parse::<String>()
        .unwrap();
    let num_threads = matches
        .get_one::<String>("num_threads")
        .unwrap()
        .parse::<usize>()
        .unwrap_or(16);
    let backlog = matches
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

    // -----------------------------------------------------------------
    // Get the appropriate image transformation
    let mut image_transform_config = None;
    if crop_and_resize {
        image_transform_config = Some(image_processing::ImageTransformConfig {
            crop_and_resize: true,
            default_image_size: 1024,
            downsampling_ratio: 32,
            min_aspect_ratio: 0.5,
            max_aspect_ratio: 2.0,
            pre_encode_images: true,
        });
    }

    // Define the source configuration, very rough as of now
    let source_config = generator_http::SourceDBConfig {
        sources: source,
        sources_ne: "".to_string(),
        require_images: true,
        require_embeddings: false,
        tags: "".to_string(),
        tags_ne: "".to_string(),
        has_attributes: "".to_string(),
        lacks_attributes: "".to_string(),
        has_masks: "".to_string(),
        lacks_masks: "".to_string(),
        has_latents: "".to_string(),
        lacks_latents: "".to_string(),
        min_short_edge: 0,
        max_short_edge: 10000,
        min_pixel_count:-1,
        max_pixel_count:-1,
        duplicate_state:-1,
        random_sampling:false,
    };

    let max_connections = 128;
    let mut client = client::DatagoClient::new(
        source_config,
        image_transform_config,
        num_samples,
        page_size,
        0,1,
        num_threads,
        max_connections,
        backlog,
    );

    let mut size_buckets: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    let start_time = std::time::Instant::now();

    client.start();
    for _ in 0..num_samples {
        if let Some(sample) = client.get_sample() {
            if crop_and_resize {
                // Count the number of samples per size
                let size = format!("{}x{}", sample.image.width, sample.image.height);
                let count = size_buckets.entry(size).or_insert(0);
                *count += 1;
            }
        } else {
            println!("Failed to get a sample");
        }
    }
    client.stop();
    println!("All samples processed");

    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let fps = num_samples as f64 / elapsed_secs;

    let mut table = Table::new();
    table.add_row(row!["Total Samples", num_samples]);
    table.add_row(row!["Execution time", format!("{:.2}", elapsed_secs)]);
    table.add_row(row!["Samples / s", format!("{:.2}", fps)]);
    table.add_row(row!["Active Threads", num_threads]);
    table.add_row(row!["Max Connections", max_connections]);
    table.printstd();
}
