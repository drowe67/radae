# Embedding Python in C

Goal: to make the existing RADAE Python implementation a C library

Approaches:

1. Cython: docs suggest the top level must be a Python program
2. [Embedding](https://docs.python.org/3/extending/embedding.html): C main() call Python functions. Currently testing this approach 
3. libtorch: worth exploring

# Packages

`sudo apt install python3.10-dev`

# Build and Run docs.python.org demo

```
gcc embed1.c -o embed1 $(python3.10-config --clags) $(python3.10-config --ldflags --embed)
PYTHONPATH="." ./embed1 mult multiply 2 2
```

