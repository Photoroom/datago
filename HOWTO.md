## Install deps

```bash
$ sudo apt install golang libjpeg-turbo8-dev libvips-dev
$ sudo ldconfig
```

## Build a benchmark CLI

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

## Run the go test suite

From the src folder

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

then from the /pkg/client folder:

```bash
$ gopy pkg -author="Photoroom" -email="team@photoroom.com" -url="" -name="datago" -version="0.0.1" .
```

then you can `pip install -e .` from here.

## Use the package from Python

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

client.Start()  # This will load up CPU and take ram until the pre-fetching is done

for _ in range(10):
    sample = client.GetSample() # This will warn and return an empty sample if the client was not Start()
```

Please note that the image buffers will be passed around as raw pointers, they can be re-interpreted in python with the attached helpers

## Update the pypi release (maintainers)
```
python3 setup.py sdist
python3 -m twine upload dist/* --verbose
```