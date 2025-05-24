"""
   Autocorrelation tool for ML OFDM fine timing experiments. Complex
   .f32 time domain samples as input, autocorrelation vectors on output
   for use as features in training a ML network.
   
/* Copyright (c) 2025 David Rowe */
   
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

# TODO need random delta, also save delta to a file

import sys,struct
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('y', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('Ry', type=str, help='path to autocorrelation output feature file dim (Ncp+M) .f32 format')
parser.add_argument('delta', type=str, help='path to delta output file dim .f32 format')
parser.add_argument('tau', type=int, help='autocorrelation lag (e.g. M or 2(Ncp+M))')
parser.add_argument('N', type=int, help='number of samples to correlate over (e.g. Ncp or M)')
parser.add_argument('-Q', type=int, default=10, help='number of past symbols to correlate over (default 10)')
parser.add_argument('-M', type=int, default=128, help='length of symbol in samples without cyclic prefix (default 128)')
parser.add_argument('-Ncp', type=int, default=32, help='length of cyclic prefix in samples (default 32)')
args = parser.parse_args()
M = args.M
Ncp = args.Ncp
tau = args.tau
N = args.N
Q = args.Q

y = np.fromfile(args.y, dtype=np.complex64)
Nsyms = len(y) // (Ncp+M)
print(Nsyms)

Ry = np.zeros(Ncp+M,dtype=np.float32)
f_Ry = open(args.Ry,"wb")
f_delta = open(args.delta,"wb")

for s in np.arange(args.Q+1,Nsyms-1):

   # random timing offset
   delta = int(np.random.random()*(Ncp+M))
   
   for delta_hat in np.arange(Ncp+M):
      Ry[delta_hat] = 0.0
      for q in np.arange(Q):
         st = (s-q)*(Ncp+M) - delta + delta_hat
         y1 = y[st-tau:st-tau+N]
         y2 = y[st:st+N]
         num = np.abs(np.dot(y1, np.conj(y2)))
         den = np.abs((np.dot(y1, np.conj(y1)) + np.dot(y2, np.conj(y2))))
         Ry[delta_hat] += num/den
   Ry.tofile(f_Ry)
   np.array([delta], dtype=np.float32).tofile(f_delta)

f_Ry.close()
f_delta.close()
