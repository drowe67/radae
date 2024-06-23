"""

Generates a .f32 IQIQI chirp file for HF channel SNR measurement

test usage:
  $ python3 chirp.py t.f32 4
  $ play -r 8k -e float -b 32 -c 2 t.f32

/* Copyright (c) 2024 David Rowe */
   
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

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('f32', type=str, help='path to output IQ .f32 file')
parser.add_argument('Nsec', type=float, default=4.0, help='output file length in seconds')
parser.add_argument('--flow', type=float, default=400.0, help='lower freq limit for chirp (default 400 Hz)')
parser.add_argument('--fhigh', type=float, default=2000.0, help='upper limit for chirp (default 2000 Hz)')
parser.add_argument('--amp', type=float, default=0.25, help='magnitude of chirp (default 0.25)')
args = parser.parse_args()

Fs = 8000
Nsam = int(args.Nsec*Fs)
x = np.zeros(Nsam, dtype=np.csingle)
freq = args.flow/Fs
delta_freq = (args.fhigh - args.flow)/Fs
phase = 0

for n in np.arange(Nsam):
    phase += 2*np.pi*freq/Fs
    phase -= 2*np.pi*int(phase/(2*np.pi))
    freq += delta_freq
    if freq > args.fhigh:
        delta_freq = -(args.fhigh - args.flow)/Fs
    if freq < args.flow:
        delta_freq = (args.fhigh - args.flow)/Fs
    x[n] = args.amp*np.exp(1j*phase)

x.tofile(args.f32)
