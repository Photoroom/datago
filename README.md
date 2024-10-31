[![Build & Test](https://github.com/Photoroom/datago/actions/workflows/go.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/go.yml)
[![Gopy](https://github.com/Photoroom/datago/actions/workflows/gopy.yml/badge.svg)](https://github.com/Photoroom/datago/actions/workflows/gopy.yml)

# datago

A golang-based data loader which can be used from Python. Compatible with a soon-to-be open sourced VectorDB-enabled data stack, which exposes HTTP requests.

Datago handles, outside of the Python GIL

- per sample IO from object storage
- deserialization (jpg and png decompression)
- some optional vision processing (aligning different image payloads)
- optional serialization

Samples are exposed in the Python scope as python native objects, using PIL and Numpy base types.
Speed will be network dependent, but GB/s is typical.

Datago is rank and world-size aware, in which case the samples are dispatched depending on the samples hash.

<img width="922" alt="Screenshot 2024-09-24 at 9 39 44 PM" src="https://github.com/user-attachments/assets/b58002ce-f961-438b-af72-9e1338527365">

<details> <summary><strong>Use it</strong></summary>

## Use the package from Python

```python
from datago import datago

config = datago.GetDefaultConfig()
# Check out the config fields, plenty of option to specify your DB query and optimize performance

client = datago.GetClient(config)
client.Start()  # This can be done early for convenience, not mandatory (can fetch samples while models are instanciated for intance)

for _ in range(10):
    sample = client.GetSample() # This start the client if not previously done, in that case latency for the first sample is higher
```

Please note that the image buffers will be passed around as raw pointers, they can be re-interpreted in python with the attached helpers

## Match the raw exported buffers with typical python types

See helper functions provided in `polyglot.py`, should be self explanatory

</details><details> <summary><strong>Build it</strong></summary>

## Install deps

```bash
$ sudo apt install golang libjpeg-turbo8-dev libvips-dev
$ sudo ldconfig
```

## Build a benchmark CLI

From the root of this project:

```bash
$ go build cmd/main/main.go
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
