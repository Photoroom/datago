# Datago Performance Optimization Worklog

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
- **Change**: Modified `src/worker_wds.rs` to use `tokio::task::spawn_blocking` for CPU-bound `image::load_from_memory` calls
- **Rationale**: The image decoding is CPU-intensive and synchronous. Using `spawn_blocking` offloads this work to tokio's blocking thread pool, allowing the async runtime to continue scheduling other tasks (network I/O, tar extraction) concurrently
- **Files Modified**: `src/worker_wds.rs`
- **Benchmark Results** (PD12M dataset, limit=100, 8 workers):
  - **Before**: 50.09 FPS
  - **After**: 53.13 FPS
  - **Improvement**: +6.1% (+3.04 FPS)
  - **Webdataset lib**: 2.17 FPS
  - **Speedup**: ~24.5x faster

---

## Baseline Performance (Before Optimizations)
| Config | Datago FPS | Webdataset FPS | Speedup |
|--------|-----------|----------------|---------|
| limit=10, workers=2 | 14.66 | 1.18 | ~12.4x |
| limit=50, workers=4 | 35.22 | 2.97 | ~11.9x |
| limit=100, workers=8 | 50.09 | 1.87 | ~26.8x |

---

## Optimization Opportunities (To Investigate)

### High Priority
1. **Image decoding parallelization** - Currently `image::load_from_memory` is synchronous and blocks the async task. Could use `spawn_blocking` for CPU-bound work.
2. **Tokio runtime consolidation** - Multiple tokio runtimes are created (one in generator_wds.rs, one in worker_wds.rs). Could share a single runtime.
3. **Buffer size tuning** - Current `prefetch_buffer_size=256`, `samples_buffer_size=256` may not be optimal.

### Medium Priority
4. **Faster image decoder** - The `image` crate may not be the fastest. Could try `rav1e` for JPEG or specialized decoders.
5. **Tarball streaming optimization** - Current async_tar + BufReader could be tuned for better throughput.
6. **Memory reuse** - Image buffers are created and dropped repeatedly. Could use object pools.

### Low Priority
7. **Network connection pooling** - Already implemented via `SharedClient` with semaphore.
8. **HTTP/2 support** - Could enable for better multiplexing.

---

## Best Speed Attained
- **Current Best**: 53.13 FPS (limit=100, workers=8)
- **Date**: 2026-05-11
- **Config**: PD12M dataset, 8 workers, concurrent_downloads=32, spawn_blocking optimization
- **Improvement from baseline**: +6.1%

### Rust Side (src/)
- [ ] Review `generator_wds.rs` - webdataset source generation
- [ ] Review `worker_wds.rs` - webdataset worker implementation  
- [ ] Review `client.rs` - main client logic
- [ ] Review async/tokio patterns for I/O
- [ ] Review buffer sizes and prefetch strategies
- [ ] Review tar/async-tar decoding performance
- [ ] Review image decoding (image crate vs alternatives)

### Python Side (python/)
- [ ] Review `dataset.py` - DatagoIterDataset wrapper
- [ ] Review buffer configurations in benchmark
- [ ] Review concurrent_downloads parameter tuning

---

## Best Speed Attained
- **Current Best**: 35.22 FPS (limit=50, workers=4)
- **Date**: 2026-05-11
- **Config**: PD12M dataset, 4 workers, concurrent_downloads=32

---
