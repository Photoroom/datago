# datago

[![Rust](https://github.com/Photoroom/datago/actions/workflows/rust.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/rust.yml)
[![Rust-py](https://github.com/Photoroom/datago/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/ci-cd.yml)

A Rust-written data loader which can be used from Python. Compatible with a [soon-to-be open sourced](https://github.com/Photoroom/dataroom) VectorDB-enabled data stack, which exposes HTTP requests, and with a local filesystem, more front-ends are possible. Focused on image data at the moment, could also easily be more generic.

Datago handles, outside of the Python GIL

- per sample IO
- deserialization (jpg and png decompression)
- some optional vision processing (aligning different image payloads)
- optional serialization

Samples are exposed in the Python scope as python native objects, using PIL and Numpy base types.
Speed will be network dependent, but GB/s is typical.

Depending on the front ends, datago can be rank and world-size aware, in which case the samples are dispatched depending on the samples hash. Only an iterator is exposed at the moment, but a map interface wouldn't be too hard.

<img width="922" alt="Screenshot 2024-09-24 at 9 39 44 PM" src="https://github.com/user-attachments/assets/b58002ce-f961-438b-af72-9e1338527365">

<details> <summary><strong>Use it</strong></summary>

~Using Python 3.11, you can simply install datago with `pip install datago`~
See https://github.com/Photoroom/datago/issues/83, needs fixing

## Use the package from Python

```python
from datago import DatagoClient
import os
import json

config = {
    "source_config": {
        "sources": os.environ.get("DATAROOM_TEST_SOURCE", ""),
        "page_size": 500,
    },
    "limit": 200,
    "rank": 0,
    "world_size": 1,
    "samples_buffer_size": 32,
}

client = DatagoClient(json.dumps(config))

for _ in range(10):
    sample = client.get_sample()
```

Please note that the image buffers will be passed around as raw pointers, see below.
To test datago while serving local files (jpg, png, ..), code would look like the following

```python
from datago import DatagoClient
import os
import json

config = {
    "source_type": "file",
    "source_config": {
        "root_path": "myPath",
    },
    "limit": 200,
    "rank": 0,
    "world_size": 1,
    "samples_buffer_size": 32,
}

client = DatagoClient(json.dumps(config))

for _ in range(10):
    sample = client.get_sample()
```


## Match the raw exported buffers with typical python types

See helper functions provided in `raw_types.py`, should be self explanatory. Check python benchmarks for examples.

</details><details> <summary><strong>Build it</strong></summary>

## Preamble

Just install the rust toolchain via rustup

## Build a benchmark CLI
`cargo run --release --  -h` to get all the information, should be fairly straightforward

## Run the rust test suite

From the datago folder

```bash
cargo test
```

## Generate the python package binaries manually

```bash
maturin build -i python3.11 --release --target "x86_64-unknown-linux-gnu"
```

then you can `pip install` from `target/wheels`

## Update the pypi release (maintainers)

Create a new tag and a new release in this repo, a new package will be pushed automatically.

</details>

# License

MIT License

Copyright (c) 2024 Photoroom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
