import torch
import numpy as np

print("some init-time numpy")
x = np.ones((3))

def multiply(a,b,arr_in,arr_out):
    print("numpy vector from init:",x, x.dtype)

    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b

    # try a little pytorch
    t = torch.ones(1,3)
    print("torch tensor:",t)

    # input array from C
    print("arr_in:",arr_in,arr_in.dtype)

    # return array to C by modifying arr_out
    # a big hack ... but not sure how to return numpy arrays
    print("arr_out:",arr_out,arr_out.dtype)
    # note we can't just assign, we need to use same memory as arr_out
    np.copyto(arr_out,x)
    
    return c

