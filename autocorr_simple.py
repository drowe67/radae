"""
/*
  autocorr_simple.py

  Simple autocorrelation tool, for testing adasmooth.m FT est

  Copyright (c) 2025 by David Rowe */

/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os,sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf
from models_ft import ftDNNXcorr
from models_sync import FrameSyncNet

parser = argparse.ArgumentParser()

parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('write_Ry', type=str, default="", help='path to autocorrelation output feature file dim (seq_len,Ncp+M) .f32 format')
parser.add_argument('-M', type=int, default=128, help='length of symbol in samples without cyclic prefix (default 128)')
parser.add_argument('--Ncp', type=int, default=32, help='length of cyclic prefix in samples (default 32)')
parser.set_defaults(bpf=True)
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
args = parser.parse_args()
M = args.M
Ncp = args.Ncp
Fs = 8000

# load rx rate_Fs samples
rx = np.fromfile(args.rx, dtype=np.csingle)
# BPF to remove some of the noise and improve acquisition
Ntap = 0
if args.bpf:
   Ntap=101
   bandwidth = 800
   centre = 1500
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   bpf = complex_bpf(Ntap, Fs, bandwidth,centre)
   rx = bpf.bpf(rx)

sequence_length = len(rx)//(Ncp+M) - 2
print(sequence_length)

Ry_norm = np.zeros((sequence_length,Ncp+M),dtype=np.complex64)
for s in np.arange(sequence_length):
   for delta_hat in np.arange(Ncp+M):
      st = (s+1)*(Ncp+M) + delta_hat
      y_cp = rx[st-Ncp:st]
      y_m = rx[st-Ncp+M:st+M]
      Ry = np.dot(y_cp, np.conj(y_m))
      D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
      Ry_norm[s,delta_hat] = 2.*Ry/abs(D)
Ry_norm.flatten().tofile(args.write_Ry)
