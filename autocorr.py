"""
   Autocorrelation tool for ML OFDM fine timing experiments. Complex
   .f32 time domain samples as input, autocorrelation vectors on output
   for use as features in training a ML network.
   
   Example:

     ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \\
      --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \\
      --auxdata --w1_dec 128 --write_rx all_rx.f32

     python3 autocorr.py rx.f32 Ry.f32 delta.f32

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
from radae import complex_bpf

parser = argparse.ArgumentParser()

parser.add_argument('y', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('Ry', type=str, help='path to autocorrelation output feature file dim (Ncp+M) .f32 format')
parser.add_argument('delta', type=str, help='path to fine timing ground truth (delta) output file dim .f32 format')
parser.add_argument('-Q', type=int, default=1, help='number of past symbols to correlate over (default 1)')
parser.add_argument('--sequence_length', type=int, default=50, help='sequence length - number of consecutive symbols with same fine timing (default 50)')
parser.add_argument('-M', type=int, default=128, help='length of symbol in samples without cyclic prefix (default 128)')
parser.add_argument('--Ncp', type=int, default=32, help='length of cyclic prefix in samples (default 32)')
parser.add_argument('--Nseq', type=int, default=0, help='extract just first Nseq sequences (default extract all)')
parser.add_argument('--bpf',  type=int, default=0, help='enable band pass filter (default off)')
parser.add_argument('--snr', type=float, default=100.00, help='Inject noise at a single SNR (default no noise)')
parser.add_argument('--range_snr', action='store_true', help='Inject noise using a range of SNRs for training (default no noise)')
parser.add_argument('--seq_hop', type=int, default=1, help='How many input symbols to jump for each training sequence (default 1)')
parser.add_argument('--test', action='store_true', help='Test mode, check argmax(Ry) == delta')
parser.add_argument('--recursion', action='store_true', help='Use efficient recursion')
args = parser.parse_args()
M = args.M
Ncp = args.Ncp
Q = args.Q
sequence_length = args.sequence_length
seq_hop = args.seq_hop
Fs = 8000
passes = 0

y = np.fromfile(args.y, dtype=np.complex64)
Nseq_available = len(y) // (Ncp+M)
if args.Nseq == 0:
   Nseq = Nseq_available
else:
   Nseq = args.Nseq

# make sure we don't run off the end of the file
Nseq_required = Nseq*seq_hop + sequence_length+Q+1
print(f"Nseq: {Nseq:d} Nseq_available: {Nseq_available:d} Nseq_required: {Nseq_required:d}")
if Nseq_required > Nseq_available:
   print(f"Nseq too large ... quiting")
   quit()

# measure signal power for entire vector to ensure a good mean if fading is present
S = (np.dot(y,np.conj(y))/len(y)).real
#S=np.var(y)
print(f"Signal power S: {S:5.2f}")

# generate a unit power complex gaussian noise vector of the samme length as y
rng = np.random.default_rng(42)
n = (rng.standard_normal(size=len(y),dtype=np.float32) + 1j*rng.standard_normal(size=len(y),dtype=np.float32))/np.sqrt(2)
print(f"var(n): {np.var(n):5.2f}")

f_Ry = open(args.Ry,"wb")
f_delta = open(args.delta,"wb")

# note we BPF noise, rather than signal+noise for convenience, this
# neatly avoids time shift due to filter delay that would shift delta
if args.bpf != 0:
   Ntap=101
   bandwidth = args.bpf
   centre = 1500
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   bpf = complex_bpf(Ntap, Fs, bandwidth, centre)
   n = bpf.bpf(n)

for seq in np.arange(Nseq):
   print(f"\r{seq:d} ", end='')

   # single random timing offset for entire sequence
   delta = int(rng.random()*(Ncp+M))

   # chunk of input samples that we add noise to
   y_ = np.copy(y[seq*seq_hop*(Ncp+M):(seq*seq_hop+sequence_length+Q+1)*(Ncp+M)])

   # SNR value for sequence
   if args.range_snr:
      SNR3kdB = -2 +  15*rng.random()
   else:
      SNR3kdB = args.snr
   SNR3k = 10**(SNR3kdB/10)
   sigma = ((S*Fs)/(SNR3k*3000))**0.5
   n_ = sigma*n[seq*seq_hop*(Ncp+M):(seq*seq_hop+sequence_length+Q+1)*(Ncp+M)]
   y_ += n_
   print(f"SNRdB: {SNR3kdB:5.2f} sigma: {sigma:5.2f} sigma_: {np.var(n_):5.2f}",end='')
   
   # calculate Ry for every time step in the sequence, plus Q-1 extra to support smoothing
   Ry_norm = np.zeros((sequence_length+Q-1,Ncp+M),dtype=np.float32)

   for s in np.arange(sequence_length+Q-1):
      if args.recursion:
         # note: this didn't speed up Python code but might be useful for C version
         # compute delta_hat = 0 the long way
         delta_hat = 0
         st = Ncp + (s+1)*(Ncp+M) - delta + delta_hat
         y_cp = y_[st-Ncp:st]
         y_m = y_[st-Ncp+M:st+M]
         Ry = np.dot(y_cp, np.conj(y_m))
         D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
         Ry_norm[s,delta_hat] = 2.*np.abs(Ry)/np.abs(D)
         # use recursion for remaining delta_hat
         for delta_hat in np.arange(1,Ncp+M):
            st = Ncp + (s+1)*(Ncp+M) - delta + delta_hat
            y_cp = y_[st-Ncp:st]
            y_m = y_[st-Ncp+M:st+M]
            #Ry = np.dot(y_cp, np.conj(y_m))
            Ry = Ry - y_[st-Ncp-1]*np.conj(y_[st-Ncp+M-1]) + y_[st-1]*np.conj(y_[st+M-1])
            D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
            Ry_norm[s,delta_hat] = 2.*np.abs(Ry)/np.abs(D)

      else:
         for delta_hat in np.arange(Ncp+M):
            # Ncp term at start to adjust input samples to y(0) time reference used in paper
            # st is the y(0) sample of the s-th symbol
            st = Ncp + (s+1)*(Ncp+M) - delta + delta_hat
            
            y_cp = y_[st-Ncp:st]
            y_m = y_[st-Ncp+M:st+M]
            
            Ry = np.dot(y_cp, np.conj(y_m))
            D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
            
            Ry_norm[s,delta_hat] = 2.*np.abs(Ry)/np.abs(D)            
            

   # Now output Ry & delta for each step in sequence, smoothing Ry over last Q symbols
   for s in np.arange(sequence_length):
      Ry_bar = np.mean(Ry_norm[s:s+Q,:],axis=0)
      Ry_bar.tofile(f_Ry)
      np.array([delta], dtype=np.float32).tofile(f_delta)
      if args.test:
         max_delta_hat = np.argmax(Ry_bar)
         nmax = Ncp + M
         # allow +/2 error, and recall error is modulo nmax e.g. error(0,159) is 1.
         # A test mode based on variance of ft_error might be useful here
         ft_error = ((max_delta_hat - delta + nmax/2) % nmax) - nmax/2
         if abs(ft_error) < 3:
            passes += 1
         else:
            print(f" delta: {delta:d} max_delta_hat: {max_delta_hat:d} ft_error: {ft_error:f}")

f_Ry.close()
f_delta.close()
print(f"")

if args.test:
   print(f"target: {Nseq*sequence_length:d} passes: {passes:d}")
   if passes == Nseq*sequence_length:
      print("PASS")
   else:
      print("FAIL")
