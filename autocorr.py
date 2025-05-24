"""
   Autocorrelation tool for ML OFDM fine timing experiments. Complex
   .f32 time domain samples as input, autocorrelation vectors on output
   for use as features in training a ML network.
   
   Example:

     ./inference.sh 250506/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null --bottleneck 3
      --rate_Fs --auxdata --EbNodB 100 --cp 0.004 --write_rx rx.f32

     python3 autocorr.py rx.f32 Ry.f32 delta.f32 -128 32

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
parser.add_argument('delta', type=str, help='path to fine timing ground truth (delta) output file dim .f32 format')
parser.add_argument('tau', type=int, help='autocorrelation lag (e.g. M or 2(Ncp+M))')
parser.add_argument('N', type=int, help='number of samples to correlate over (e.g. Ncp or M)')
parser.add_argument('--sequence_length', type=int, default=50, help='sequence length - number of consectutive symbols with same fine timing (default 100)')
parser.add_argument('-M', type=int, default=128, help='length of symbol in samples without cyclic prefix (default 128)')
parser.add_argument('-Ncp', type=int, default=32, help='length of cyclic prefix in samples (default 32)')
args = parser.parse_args()
M = args.M
Ncp = args.Ncp
tau = args.tau
N = args.N
sequence_length = args.sequence_length

y = np.fromfile(args.y, dtype=np.complex64)
Nseq = len(y) // (Ncp+M) - sequence_length - 1
print(f"Nseq: {Nseq:d}")

Ry = np.zeros(Ncp+M,dtype=np.float32)
f_Ry = open(args.Ry,"wb")
f_delta = open(args.delta,"wb")

for seq in np.arange(Nseq):

   # random timing offset for sequence
   delta = int(np.random.random()*(Ncp+M))
   
   for s in np.arange(sequence_length):
      for delta_hat in np.arange(Ncp+M):
         # note overlapping sequences with different delta for data augmentation
         st = (seq+s+1)*(Ncp+M) - delta + delta_hat
         y1 = y[st-tau:st-tau+N]
         y2 = y[st:st+N]
         num = np.abs(np.dot(y1, np.conj(y2)))
         den = np.abs((np.dot(y1, np.conj(y1)) + np.dot(y2, np.conj(y2))))
         Ry[delta_hat] = num/den
      Ry.tofile(f_Ry)
      np.array([delta], dtype=np.float32).tofile(f_delta)
   print(seq)

f_Ry.close()
f_delta.close()
