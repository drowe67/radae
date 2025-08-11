"""
  Baseband simulation of analog FM using (11) from paper.

  Fs=8000 Hz int16 speech samples on stdin, output samples on stdout.

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

import sys,struct
import argparse
import numpy as np
import math as m

parser = argparse.ArgumentParser()

parser.add_argument('--RdBm', type=float, default=-100, help='Receive level set point in dBm')
parser.add_argument('--h_file', type=str, default="", help='Path to rate Fs fading channel magnitude samples, rate Fs time steps by Nc=1 carriers .f32 format')
parser.add_argument('--fading_advance', type=float, default=0, help='Where to start sampling fading samples in seconds (default 0)')
parser.add_argument('-v', action='store_true', help='Verbose debug info')
args = parser.parse_args()
RdBm = args.RdBm

Am = 16384  # peak input int16 level, corresponds to max deviation f_d 
A = 1       # normalised peak input level assumed in SNR expression
x_bar = 0.5 # for sine wave with peak A=1

k = 1.38E-23; T=274; NFdB = 5
Fs = 8000
fd_Hz = 2500
fm_Hz = 3000
beta = fd_Hz/fm_Hz
Gfm = 10*m.log10(3*(beta**2)*x_bar/(1E3*k*T*fm_Hz)) - NFdB
TdBm = 12 - Gfm

print(f"Fs: {Fs:5.2f} Deviation: {fd_Hz} Hz  Max Modn freq: {fm_Hz} Hz Beta: {beta:3.2f}", file=sys.stderr)
print(f"x_bar: {x_bar:5.2f} Gfm: {Gfm:5.2f} dB TdB: {TdBm:5.2f} dB  RdBm: {RdBm:5.2f}", file=sys.stderr)

# user supplied rate Rs multipath model, sequence of H magnitude samples
if len(args.h_file):
  H = np.fromfile(args.h_file, dtype=np.float32)
  fading_index = int(args.fading_advance*Fs)
  print(f"fading_adv: {args.fading_advance:f} offset (samples): {fading_index:d}",  file=sys.stderr)

# average noise and signal power
n2_sum = 0.0
x2_sum = 0.0
n_sum = 0
n_clipped = 0
sigma = 0.0

while True:
    buffer = sys.stdin.buffer.read(struct.calcsize("h"))
    if len(buffer) != struct.calcsize("h"):
      break
    x = np.frombuffer(buffer,np.int16).astype(np.float32)[0]

    RdBm_dash = RdBm
    if args.h_file:
      if fading_index > len(H)-1:
        print(f"ERROR; h_file too short for sample! Quitting", file=sys.stderr)
        sys.exit(1)
      RdBm_dash = 20*m.log10(H[fading_index]) + RdBm_dash
      fading_index += 1

    if RdBm_dash > TdBm:
      SNRdB = RdBm_dash + Gfm
    else:
      SNRdB = 3*RdBm_dash + Gfm - 2*TdBm

    # Work out sigma of the noise generator.  Eq (11) is the SNR in noise bandwidth f_m Hz.
    # We want to simulate at Fs Hz. The noise generator spreads noise uniformly across Fs/2 Hz.
    # After generation of the noise at any sample rate, the power in f_m Hz should be the same
    # as given by (11).  This can be achieved by keeping the noise density constant across sample
    # rate changes.  
    SNR = 10**(SNRdB/10)   # Linear SNR in fm Hz
    N_fm = x_bar/SNR       # Noise power in fm Hz
    No   = N_fm/fm_Hz      # noise density, we want to preserve this with change in sample rate
    N_Fs2 = No*Fs/2        # total noise power in Fs/2 Hz
    sigma = N_Fs2 **0.5
    sigma *= Am            # map A=1 to int16 peak
    if args.v:
        print(f"SNRdB: {SNRdB:5.2f}", file=sys.stderr)
    n = sigma*np.random.randn()

    n2_sum += n*n
    x2_sum += x*x
    n_sum += 1
    x += n
    if x > 32767.0:
        x = 32767.0
        n_clipped += 1
    if x < -32776.0:
        x = -32776.0
        n_clipped += 1

    x = np.float32(x).astype(np.int16)
    sys.stdout.buffer.write(x) 

SNRdB_ = 10.0*m.log10(x2_sum/n2_sum)
x_bar_ = x2_sum/(n_sum*Am*Am)
percent_clipped = 100 * n_clipped/n_sum
print(f"SNRdB setpoint: {RdBm+Gfm:5.2f} SNRdB_ measured: {SNRdB_:5.2f} SNRdB_ - SNRdB: {SNRdB_-SNRdB:5.2f}", file=sys.stderr)
print(f"x_bar measured: {x_bar_:5.2f} %clipped {percent_clipped:5.2f} sigma: {sigma:5.2f}", file=sys.stderr)
