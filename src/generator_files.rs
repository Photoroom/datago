use serde::{Deserialize, Serialize};
use std::hash::Hash;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileConfig {
    pub root_path: String,
}

// Hash function to be able to dispatch the samples to the correct rank
fn hash<T: Hash>(t: &T) -> u64 {
    use std::hash::Hasher;
    let mut s = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub fn ping_files(
    pages_tx: kanal::Sender<serde_json::Value>,
    source_config: SourceFileConfig,
    rank: usize,
    world_size: usize,
    num_samples: usize,
) {
    // Get an iterator over the files in the root path
    let supported_extensions = [
        "jpg", "jpeg", "png", "bmp", "gif", "JPG", "JPEG", "PNG", "BMP", "GIF",
    ];

    let files = walkdir::WalkDir::new(&source_config.root_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok());

    let files = files.filter_map(|entry| {
        let path = entry.path();
        let file_name = path.to_str().unwrap().to_string();
        if supported_extensions
            .iter()
            .any(|&ext| file_name.ends_with(ext))
        {
            Some(entry)
        } else {
            None
        }
    });

    let page_size = 500;

    // While we have something in the Send the samples to the channel
    let mut count = 0;
    let mut filepaths = Vec::new();

    // Build a page from the files iterator
    for entry in files {
        let file_name = entry.path().to_str().unwrap().to_string();

        // If world_size is not 0, we need to dispatch the samples to the correct rank
        if world_size > 1 {
            let hash = hash(&file_name);
            let target_rank = (hash % world_size as u64) as usize;
            if target_rank != rank {
                continue;
            }
        }

        filepaths.push(file_name);
        count += 1;

        if filepaths.len() >= page_size || count >= num_samples {
            // Convert the page to a JSON
            let page_json = serde_json::json!({
                "results": filepaths,
                "rank": rank,
                "world_size": world_size,
            });

            // Push the page to the channel
            if pages_tx.send(page_json).is_err() {
                println!("ping_pages: stream already closed, wrapping up");
                break;
            }
            filepaths.clear();
        }

        if count >= num_samples {
            // NOTE: This doesn´t count the samples which have actually been processed
            println!("ping_pages: reached the limit of samples requested. Shutting down");
            break;
        }
    }

    // Either we don't have any more samples or we have reached the limit
    println!(
        "ping_pages: total samples requested: {}. page samples served {}",
        num_samples, count
    );

    // Send an empty value to signal the end of the stream
    match pages_tx.send(serde_json::Value::Null) {
        Ok(_) => {}
        Err(_) => {
            println!("ping_pages: stream already closed, all good");
        }
    };
}
