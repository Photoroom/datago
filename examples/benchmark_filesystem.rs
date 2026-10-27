//! Native baseline for python/benchmark_filesystem.py. Run with --release.

use clap::{value_parser, Arg, ArgAction, Command};
use datago::DatagoClient;
use serde_json::json;
use std::{error::Error, hint::black_box, num::NonZeroUsize, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args = Command::new("benchmark_filesystem")
        .about("Measure native Datago filesystem throughput without entering Python")
        .arg(
            Arg::new("root")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .default_value("2000")
                .value_parser(value_parser!(NonZeroUsize)),
        )
        .arg(
            Arg::new("workers")
                .long("num-workers")
                .help("In-flight tasks; defaults to CPU count, 0 means serial")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("resize")
                .long("crop-and-resize")
                .help("Resize and center-crop to 1024 x 1024, matching the Python benchmark")
                .action(ArgAction::SetTrue),
        )
        .get_matches();
    let root = args.get_one::<PathBuf>("root").expect("required argument");
    if !root.is_dir() {
        return Err(format!("not a directory: {}", root.display()).into());
    }
    let limit = args
        .get_one::<NonZeroUsize>("limit")
        .expect("default limit")
        .get();
    let workers = args
        .get_one::<usize>("workers")
        .copied()
        .unwrap_or_else(num_cpus::get);
    let resize = args.get_flag("resize");
    std::env::set_var("DATAGO_MAX_TASKS", workers.to_string());
    let mut config = json!({
        "source_type": "file",
        "source_config": {"root_path": root, "rank": 0, "world_size": 1},
        "samples_buffer_size": 256,
        "limit": limit,
    });
    if resize {
        config["image_config"] = json!({
            "crop_and_resize": true, "default_image_size": 1024,
            "downsampling_ratio": 32, "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 1.0, "pre_encode_images": false,
        });
    }

    let start = Instant::now();
    let client = DatagoClient::new(config.to_string());
    client.start();
    let (mut count, mut pixels, mut bytes) = (0_u64, 0_u64, 0_u64);
    let mut first_sample = None;
    for _ in 0..limit {
        let Some(sample) = client.get_sample() else {
            break;
        };
        first_sample.get_or_insert_with(|| start.elapsed().as_secs_f64());
        let image = sample.image.as_ref();
        count += 1;
        pixels += image.width as u64 * image.height as u64;
        bytes += image.data.len() as u64;
        black_box(sample);
    }
    let elapsed = start.elapsed().as_secs_f64();
    client.stop();
    if count != limit as u64 {
        return Err(format!("expected {limit} samples, received {count}").into());
    }
    let startup = first_sample.expect("positive limit and complete run");
    let round = |value: f64| (value * 1000.0).round() / 1000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": "rust", "image_format": "raw", "num_workers": workers,
            "crop_and_resize": resize, "count": count, "pixels": pixels, "bytes": bytes,
            "elapsed_seconds": round(elapsed), "startup_seconds": round(startup),
            "steady_fps": if count > 1 { Some(round((count - 1) as f64 / (elapsed - startup))) } else { None },
            "fps": round(count as f64 / elapsed),
            "megapixels_per_second": round(pixels as f64 / elapsed / 1e6),
            "bandwidth_mbps": round(bytes as f64 / elapsed / 1e6),
        }))?
    );
    Ok(())
}
