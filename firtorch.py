import numpy as np
import torch
from torch import nn
import os,argparse

ntaps=102
nbuf=80
fir = nn.Conv1d(1, 1, kernel_size=ntaps, padding='valid',bias=False,device="cpu")
# octave:17> Fs=8000; filt_n=101;  bpf_coeff=fir2(filt_n,[0 250 350 2500 2700 4000]/(Fs/2),[0.001 0.001 1 1 0.001 0.001]);
fir.weight.data[0,0,:] = torch.tensor([0.000097,0.000148,0.000337,0.000264,0.000286,0.000457,0.000377,0.000328,0.000393,0.000242,0.000083,-0.000130,-0.000384,-0.000597,-0.001232,-0.001541,-0.001452,-0.002560,-0.002719,-0.001647,-0.003107,-0.002973,0.000068,-0.001699,-0.001447,0.004440,0.002211,0.001509,0.010851,0.007493,0.003779,0.016633,0.011420,0.001475,0.017979,0.010512,-0.009357,0.011399,0.003054,-0.031211,-0.003712,-0.009029,-0.063631,-0.024468,-0.017229,-0.106838,-0.044317,0.001601,-0.186155,-0.056857,0.467904,0.467904,-0.056857,-0.186155,0.001601,-0.044317,-0.106838,-0.017229,-0.024468,-0.063631,-0.009029,-0.003712,-0.031211,0.003054,0.011399,-0.009357,0.010512,0.017979,0.001475,0.011420,0.016633,0.003779,0.007493,0.010851,0.001509,0.002211,0.004440,-0.001447,-0.001699,0.000068,-0.002973,-0.003107,-0.001647,-0.002719,-0.002560,-0.001452,-0.001541,-0.001232,-0.000597,-0.000384,-0.000130,0.000083,0.000242,0.000393,0.000328,0.000377,0.000457,0.000286,0.000264,0.000337,0.000148,0.000097])

parser = argparse.ArgumentParser()
parser.add_argument('file_in', type=str, help='path to input int16 sample file')
parser.add_argument('file_out', type=str, help='path to output int16 sample file')
args = parser.parse_args()

samples_in = torch.tensor(np.fromfile(args.file_in, dtype=np.int16).reshape(1,1,-1), dtype=torch.float32)
print(samples_in.shape)
samples_out = fir(samples_in)
samples_out = np.array(samples_out.detach().numpy(), dtype=np.int16)
print(samples_out.shape)
samples_out.tofile(args.file_out)
