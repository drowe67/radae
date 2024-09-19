import torch
import numpy as np

print("init code")
x = np.ones((1,2))

def multiply(a,b,arr):
    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b
    t = torch.ones(1,3)
    print("numpy vector:",x)
    print("torch tensor:",t)
    print("arr:",arr,arr.dtype)
    return c

