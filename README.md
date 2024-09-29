![Build & Test](https://github.com/github/docs/actions/workflows/go.yml/badge.svg)

datago
======

A golang-based data loader which can be used from Python. Compatible with a soon-to-be open sourced VectorDB-enabled data stack, which exposes HTTP requests.

Datago will handle, outside of the Python GIL
- per sample IO from object storage
- deserialization
- some optional vision processing (aligning different image payloads)
- serialization

Samples are then exposed in the Python scope and ready for consumption, typically using PIL and Numpy base types.
Speed will be network dependent, but GB/s is relatively easily possible

Datago can be rank and world-size aware, in which case the samples are dispatched depending on the samples hash.

<img width="922" alt="Screenshot 2024-09-24 at 9 39 44 PM" src="https://github.com/user-attachments/assets/b58002ce-f961-438b-af72-9e1338527365">


<details> <summary><strong>Use it</strong></summary>

Use the package from Python
---------------------------

```python
from datago import datago

# source, has/lacks attributes, has/lacks masks, has/lacks latents, metadata prefetch, sample prefetch, concurrent download
client = datago.GetClient(
            source="SOURCE",
            require_images=True,
            has_attributes="",
            lacks_attributes="",
            has_masks="",
            lacks_masks="",
            has_latents="",
            lacks_latents="",
            crop_and_resize=True,
            prefetch_buffer_size=64,
            samples_buffer_size=64,
            downloads_concurrency=64,
        )

client.Start()  # This can be done early for convenience, not mandatory (can fetch samples while models are instanciated for intance)

for _ in range(10):
    sample = client.GetSample() # This start the client if not previously done, in that case latency for the first sample is higher
```

Please note that the image buffers will be passed around as raw pointers, they can be re-interpreted in python with the attached helpers


Match the raw exported buffers with typical python types
--------------------------------------------------------

See helper functions provided in `polyglot.py`, should be self explanatory

</details><details> <summary><strong>Build it</strong></summary>

Install deps
------------

```bash
$ sudo apt install golang libjpeg-turbo8-dev libvips-dev
$ sudo ldconfig
```

Build a benchmark CLI
---------------------

From the root of this project `datago_src`:

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

Run the go test suite
---------------------

From the src folder

```bash
$ go test -v tests/client_test.go
```

Refresh the python package and its binaries
-------------------------------------------

- Install the dependencies as detailed in the next point
- Run the `generate_python_package.sh` script

Generate the python package binaries manually
---------------------------------------------

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

then from the /pkg/client folder:

```bash
$ gopy pkg -author="Photoroom" -email="team@photoroom.com" -url="" -name="datago" -version="0.0.1" .
```

then you can `pip install -e .` from here.


Update the pypi release (maintainers)
-------------------------------------
```
python3 setup.py sdist
python3 -m twine upload dist/* --verbose
```
</details>


License
=======
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
