# Embedding Python in C

Goal: to make the existing RADAE Python implementation a C library

Approaches:

1. Cython: docs suggest the top level must be a Python program
2. [Embedding](https://docs.python.org/3/extending/embedding.html): C main() call Python functions. Currently testing this approach. See also [Learning Python](https://learning-python.com/class/Workbook/unit16.htm) which is out of date but has examples of a cleaner API that still exists.
3. libtorch: worth exploring

# Packages

Need pythonx.y-dev so C program can fine `Python.h`, adjust for your Python version.

`sudo apt install python3.10-dev`

# Test1 - move numpy arrays C<->Python, basic numpy and PyTorch

Adapted from [2] above, basic test of numpy, torch, and moving numpy vectors between C and Python.

Building on Machine 1 (Ubuntu 20):
```
gcc embed1.c -o embed1 $(python3-config --cflags) $(python3-config --ldflags --embed) -fPIE
```
Building on Machine 2 (Ubuntu 22):
```
gcc embed1.c -o embed1 $(python3.10-config --cflags) $(python3.10-config --ldflags --embed)
```
Different build cmd lines suggests we need to focus on one distro/Python version, or have some Cmake magic to work out the gcc options.

To run:
```
PYTHONPATH="." ./embed1 mult multiply 2 2
```

# Test 2 - run RADAE in Python, C top level

This is a more serious test of running the RADAE decoder in a Python function, kicked off by a top level C program.  Requires `features_in.f32` as input (to create see many examples in Ctests, inference.sh etc).
Ubuntu 22 Build & Run:
```
gcc embed_dec.c -o embed_dec $(python3.10-config --cflags) $(python3.10-config --ldflags --embed)
PYTHONPATH="." ./embed_dec embed_dec my_decode
<snip>
Rs: 50.00 Rs': 50.00 Ts': 0.020 Nsmf: 120 Ns:   6 Nc:  20 M: 160 Ncp: 0
Processing: 972 feature vectors
loss: 0.145
```
Compare with vanilla run just from Python:
```
python3 embed_dec.py
<snip>
loss: 0.145
```
# Test 3 - radae_tx as C program

First pass at the a C callable version of `radae_tx`.  Have hard coded a few arguments for convenience, and it's a C application (rather than a library).  If this was in library form we would be ready for linking with other C applications.

Build `radae_tx` and run ctests:
```
cd radae/build
cmake ..
ctest -V -R radae_tx_embed
```

# Test 4 - radae_rx as C program

Build `radae_rx` and run ctests:
```
cd radae/build
cmake ..
ctest -V -R radae_rx_embed
```

