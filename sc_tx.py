"""
   Single Carrier Modem Tx (modulator).  Takes BBFM symbols on stdin 
   from the BBFM encoder and outputs a sequence of real int16 samples on
   stdout for input to a FM radio modulator. 

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

import sys,struct
import argparse
import numpy as np
from radae import single_carrier

parser = argparse.ArgumentParser()
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--scale', type=float, default=16384.0, help='int16 representation of float 1.0')
parser.add_argument('--ber_test', action='store_true', help='Send test frames of BPSK bits for BER test')
parser.add_argument('--fcentreHz', type=float, help='Centre frequency',default=1500)
parser.add_argument('--Rs', type=float, help='Symbol rate',default=2400)
parser.add_argument('--Fs', type=float, help='Sample rate, Fs/Rs must be an integer',default=9600)
parser.add_argument('--complex', dest="real", action='store_false', help='complex 2*int16 output samples (default real)')
parser.set_defaults(real=True)
args = parser.parse_args()

if (args.fcentreHz < args.Rs/2) and args.real and (args.fcentreHz != 0):
   print("Warning - aliased likely with real valued output samples, consider --complex")
modem = single_carrier(Rs=args.Rs, Fs=args.Fs, fcentreHz=args.fcentreHz)
assert modem.Npayload_syms == args.latent_dim

if args.ber_test:
   tx_symbs = 1 - 2*(modem.rng.random(args.latent_dim) > 0.5) + 0*1j

n_floats_in = args.latent_dim*struct.calcsize("f")
while True:
   buffer = sys.stdin.buffer.read(n_floats_in)
   if len(buffer) != n_floats_in:
      break
   z = np.frombuffer(buffer,np.float32)
   if args.ber_test:
       tx = args.scale*modem.tx(tx_symbs)
   else:
       tx = args.scale*modem.tx(z)
   if args.real:
      tx = tx.real
   tx = tx.astype(np.int16)
   sys.stdout.buffer.write(tx) 
