[![Build & Test](https://github.com/Photoroom/datago/actions/workflows/go.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/go.yml)
[![Gopy](https://github.com/Photoroom/datago/actions/workflows/gopy.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/gopy.yml)

# datago

A golang-based data loader which can be used from Python. Compatible with a [soon-to-be open sourced](https://github.com/Photoroom/dataroom) VectorDB-enabled data stack, which exposes HTTP requests, and with a local filesystem, more front-ends are possible. Focused on image data at the moment, could also easily be more generic.

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

## Use the package from Python

```python
from datago import datago
import json

client_config = {
    # two sources are supported at the moment, DB (API and stack to be shared) & local filesystem
    # in the case of the filesystem, datago will serve the jpg/png files, ID being filepath
    "source_type": datago.SourceTypeFileSystem,
    "source_config": {
        "page_size": 512,
        "root_path": root_path,
    },
    # this governs the image pre-processing, which will resize and crop to aspect ratio buckets
    # resizing is high quality by default
    "image_config": {
        "crop_and_resize": crop_and_resize,
        "default_image_size": 512,
        "downsampling_ratio": 16,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0,
        "pre_encode_images": False,
    },
    # some performance options, best settings will depend on your machine
    "prefetch_buffer_size": 64,
    "samples_buffer_size": 128,
    "concurrency": concurrency,
}

client = datago.GetClientFromJSON(json.dumps(config))
client.Start()  # This can be done early for convenience, not mandatory

for _ in range(10):
    sample = client.GetSample()
```

Please note that the image buffers will be passed around as raw pointers, see below.

## Match the raw exported buffers with typical python types

See helper functions provided in `types.py`, should be self explanatory. Check python benchmarks for examples.

</details><details> <summary><strong>Build it</strong></summary>

## Install deps

```bash
$ sudo apt install golang libjpeg-turbo8-dev libvips-dev
$ sudo ldconfig
```

## Build a benchmark CLI

From the root of this project:

```bash
$ go build cmd/main.go
```

Running it:

```bash
$ ./main --help` will tell you all about it
```

Running it with additional sanity checks

```bash
$ go run -race cmd/main/main.go
```

## Run the go test suite

From the root folder

```bash
$ go test -v tests/client_test.go
```

## Refresh the python package and its binaries

- Install the dependencies as detailed in the next point
- Run the `generate_python_package.sh` script

## Generate the python package binaries manually

```bash
$ python3 -m pip install pybindgen
$ go install golang.org/x/tools/cmd/goimports@latest
$ go install github.com/go-python/gopy@latest
$ go install golang.org/x/image/draw
```

NOTE:

- you may need to add `~/go/bin` to your PATH so that gopy is found.
- - Either `export PATH=$PATH:~/go/bin` or add it to your .bashrc
- you may need this to make sure that LDD looks at the current folder `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.`

then from the /pkg folder:

```bash
$ gopy pkg -author="Photoroom" -email="team@photoroom.com" -url="" -name="datago" -version="0.0.1" .
```

then you can `pip install -e .` from here.

## Update the pypi release (maintainers)

```
python3 setup.py sdist
python3 -m twine upload dist/* --verbose
```

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
