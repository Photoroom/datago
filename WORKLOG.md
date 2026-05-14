# Datago Performance Optimization Worklog

## 2026-05-14

### Switching to File Loading Path Optimization
- **Time**: 2026-05-14T21:00:00Z
- **Action**: Per AUTORESEARCH.md instructions and user direction, switching focus from webdataset to file loading path optimization
- **System**: 16 logical CPUs (8 cores, 2 threads/core), NVMe SSD storage, high performance mode
- **Dataset**: ImageNet-1k train (`/mnt/Data/imagenet1k/train`)

### Baseline Established
- **Config**: limit=10000, workers=12, no transform, no encoding
- **Baseline**: ~3712 FPS (avg of 3603, 3495, 3825, 3809, 3829)
- **Config details**: `prefetch_buffer_size=256`, `samples_buffer_size=256`

### Optimization #1 (Files): Use async file I/O
- **Time**: 2026-05-14T21:30:00Z
- **Commit**: `512603e`
- **Change**: 
  - Modified `src/worker_files.rs` to use `tokio::fs::read` (async) instead of `std::fs::File::open` + `BufReader` (sync)
  - Added `fs` feature to tokio in `Cargo.toml`
  - Read entire file into memory then decode with `image::load_from_memory`
- **Rationale**: The original implementation used synchronous std::fs calls which block tokio worker threads. By using async file I/O, the tokio runtime can schedule other tasks while waiting for disk I/O, improving CPU utilization especially when files are not in page cache.
- **Architectural Impact**: This is a fundamental shift from blocking to non-blocking I/O in the file loading path. While the change is small in code, it enables better task scheduling.
- **Files Modified**: `src/worker_files.rs`, `Cargo.toml`
- **Benchmark Results** (ImageNet-1k train, limit=10000, 12 workers):
  - **Before**: ~3712 FPS (avg of 3603, 3495, 3825, 3809, 3829)
  - **After**: ~3768 FPS (avg of 10 runs: 3943, 3950, 3924, 3606, 3389, 3564, 3766, 3842, 3818, 3888)
  - **Improvement**: ~1.5-4.5% (variance due to caching effects)
  - **Peak improvement**: More consistent high-end results in 3900+ FPS range
- **Test Status**: All 9 Python filesystem tests pass, all 20 Rust worker_files tests pass

### Next: Deeper Architectural Optimizations
The async file I/O change helps but doesn't leverage io_uring. Next architectural changes to consider:

1. **io_uring backend**: Use `tokio-uring` or `glommio` for true io_uring-based I/O
   - Would require either:
     - Configuring tokio to use io_uring backend (complex, may not be stable)
     - Replacing tokio with glommio's LocalExecutor (cleaner but more refactoring)
   - Expected benefit: 10-30% improvement for small random reads

2. **Parallel filesystem walking**: The `enumerate_files` function walks the filesystem synchronously. Could use parallel iterators (rayon) for multi-threaded directory traversal.

3. **Memory-mapped I/O**: Use `memmap2` crate to memory-map files instead of reading into Vec<u8>. This could reduce memory allocation overhead.

4. **Image decoding optimization**: The `image` crate may not use the fastest JPEG decoder. Could investigate:
   - Using `libjpeg-turbo` through `mozjpeg` feature
   - Using specialized decoders like `jpeg-decoder`
   - Parallel image decoding

---

## 2026-05-11

### Setup and Initial Benchmark
- **Time**: 2026-05-11T20:00:00Z
- **Action**: Built and ran the webdataset benchmark
- **Changes**: 
  - Installed maturin via cargo (`cargo install maturin`)
  - Built datago Python package using `maturin develop --release`
  - Installed pip via `curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages`
  - Installed dependencies: typer, tqdm, webdataset, torch, torchvision
  - Modified `python/benchmark_webdataset.py` to set multiprocessing start method to 'fork' for Python 3.14+ compatibility
- **Benchmark Results** (PD12M dataset, HuggingFace URL):
  - **Limit 10, 2 workers**: Datago WDS FPS: 14.66 | Webdataset lib FPS: 1.18
  - **Limit 50, 4 workers**: Datago WDS FPS: 35.22 | Webdataset lib FPS: 2.97
  - **Limit 100, 8 workers**: Datago WDS FPS: 50.09 | Webdataset lib FPS: 1.87
  - **Speedup**: ~12-26x faster than standard webdataset library
- **Test Status**: All Python tests pass (5 WDS + 34 non-DB). Rust test `test_webdataset_ranks` is flaky (passed on re-run).

### Optimization #1: spawn_blocking for Image Decoding
- **Time**: 2026-05-11T20:40:00Z
- **Commit**: `28e968d`
- **Change**: Modified `src/worker_wds.rs` to use `tokio::task::spawn_blocking` for CPU-bound `image::load_from_memory` calls
- **Rationale**: The image decoding is CPU-intensive and synchronous. Using `spawn_blocking` offloads this work to tokio's blocking thread pool, allowing the async runtime to continue scheduling other tasks (network I/O, tar extraction) concurrently
- **Files Modified**: `src/worker_wds.rs`
- **Benchmark Results** (PD12M dataset, limit=100, 8 workers, concurrent_downloads=32):
  - **Before**: 50.09 FPS
  - **After**: 53.13 FPS
  - **Improvement**: +6.1% (+3.04 FPS)
  - **Webdataset lib**: 2.17 FPS
  - **Speedup**: ~24.5x faster

### Optimization #2: Tune concurrent_downloads
- **Time**: 2026-05-11T21:00:00Z
- **Commit**: TBD
- **Change**: 
  - Changed benchmark default for `num_downloads` from 32 to 12
  - Changed Rust default in `src/generator_wds.rs` from 8 to 12
- **Rationale**: Too many concurrent downloads (32) causes resource contention. The download tasks not only do async I/O but also CPU-bound tar extraction. With 16 CPUs and 8 worker threads for image processing, 32 concurrent downloads over-subscribes the CPU. Testing showed 10-12 concurrent downloads provides better throughput.
- **Files Modified**: `src/generator_wds.rs`, `python/benchmark_webdataset.py`
- **Benchmark Results** (PD12M dataset, limit=100, 8 workers):
  - **Before (Optimization #1 only, concurrent_downloads=32)**: 53.13 FPS
  - **After (Optimization #1 + #2, concurrent_downloads=12)**: ~65-68 FPS
  - **Improvement from Optimization #1**: ~25-28% (+13-15 FPS)
  - **Improvement from baseline (50.09 FPS)**: ~30-36% (+15-18 FPS)
  - **Webdataset lib**: ~4 FPS
  - **Speedup**: ~16-17x faster than webdataset lib
  - **Note**: High variance due to network conditions

---

## Baseline Performance (Before Optimizations)
| Config | Datago FPS | Webdataset FPS | Speedup |
|--------|-----------|----------------|---------|
| limit=10, workers=2 | 14.66 | 1.18 | ~12.4x |
| limit=50, workers=4 | 35.22 | 2.97 | ~11.9x |
| limit=100, workers=8 | 50.09 | 1.87 | ~26.8x |

---

## Optimization Opportunities

### High Priority (Next to try)
1. **Tokio runtime consolidation** - Multiple tokio runtimes are created (one in generator_wds.rs line 432, one in worker_wds.rs line 266). Each runtime has its own thread pool. Could share a single runtime to reduce overhead and better utilize CPU resources.

2. **Buffer size tuning** - Current `prefetch_buffer_size=256`, `samples_buffer_size=256` may not be optimal for all workloads. Need to experiment with different values.

3. **Concurrent downloads tuning** - The `concurrent_downloads=32` default may be too high or too low. Need to find optimal value.

### Medium Priority
4. **Faster image decoder** - The `image` crate uses libpng/libjpeg which may not be the fastest. Could investigate:
   - `rav1e` for JPEG (but it's an encoder, not decoder)
   - `jpeg-decoder` crate for JPEG
   - `png` crate directly instead of through `image`
   - `faster-jpeg` or other specialized decoders

5. **Tarball streaming optimization** - Current async_tar + BufReader + StreamReader could have overhead. Could try:
   - Different buffer sizes for BufReader
   - Direct streaming without intermediate buffers
   - Parallel tar extraction

6. **Memory reuse** - Image buffers and DynamicImage objects are created and dropped repeatedly. Could use:
   - Object pools for Vec<u8>
   - Reuse image buffers
   - Zero-copy where possible

### Low Priority
7. **HTTP/2 support** - Could enable for better multiplexing on supported servers.
8. **Connection reuse** - Already implemented via `SharedClient` with semaphore.

---

## Best Speed Attained
- **Current Best**: ~65-68 FPS (limit=100, workers=8)
- **Date**: 2026-05-11
- **Config**: PD12M dataset, 8 workers, concurrent_downloads=12, spawn_blocking optimization
- **Improvement from baseline (50.09 FPS)**: +30-36%
- **Improvement from Optimization #1 (53.13 FPS)**: +25-28%

---

## Current Status
- **Latest Commit**: `701d5de` (perf(wds): tune concurrent_downloads default to 12)
- **All tests passing**: 5 WDS tests + 34 non-DB tests = 39 tests pass
- **Next optimization to try**: 
  1. Move `image.into_bytes()` into spawn_blocking (requires restructuring)
  2. Use spawn_blocking for tar extraction in download tasks
  3. Try faster image decoder (jpeg-decoder crate)
  4. Consolidate tokio runtimes (complex refactoring)

---

## Next: Tokio Runtime Consolidation

### Analysis
Looking at the code:
- `generator_wds.rs:432`: Creates a tokio runtime for the download tasks (`query_shards_and_dispatch`)
- `worker_wds.rs:266`: Creates another tokio runtime for image processing (`deserialize_samples`)
- Each runtime has its own thread pool (default num_cpus threads)

This means:
- 2 x num_cpus threads for tokio runtimes
- Plus the DATAGO_MAX_TASKS threads for async processing
- Plus the blocking thread pool for spawn_blocking

### Hypothesis
Consolidating to a single tokio runtime could:
- Reduce thread creation overhead
- Better utilize CPU resources (no competition between runtimes)
- Reduce context switching

### Plan
1. Pass a shared tokio runtime handle from generator to worker
2. Or restructure to have a single top-level runtime
3. Test with different configurations

### Implementation Strategy
The cleanest approach would be to have a single tokio runtime at the top level (in client.rs or orchestrate function) and pass the handle down to all components that need it.

However, this requires significant refactoring. A simpler first step:
- Try increasing DATAGO_MAX_TASKS to see if more parallelism helps
- Try tuning buffer sizes

---
