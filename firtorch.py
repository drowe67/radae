import numpy as np
import torch
from torch import nn
import os,argparse

ntaps=101
nbuf=80
fir = nn.Conv1d(1, 1, kernel_size=ntaps, padding='valid',bias=False,device="cpu")
fir.weight.data = 

parser = argparse.ArgumentParser()
parser.add_argument('file_in', type=str, help='path to input int16 sample file')
parser.add_argument('file_out', type=str, help='path to output int16 sample file')
args = parser.parse_args()

samples_in = torch.tensor(np.fromfile(args.file_in, dtype=np.int16).reshape(1,1,-1))
print(samples_in.shape)
samples_out = fir(samples_in).cpu().detach().numpy()
print(samples_out.shape)
samples_out.tofile(args.file_out)
