"""
   Single Carrier Modem Rx (demodulator).  Takes real int16 samples on 
   stdin from a FM demodulator, outputs frames of BBFM symbols on stdout
   to bbfm_rx.py decoder

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
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--ber_test', action='store_true', help='Receive test frames of BPSK bits for BER test')
parser.add_argument('--fcentreHz', type=float, help='Centre frequency',default=1500)
parser.add_argument('--Rs', type=float, help='Symbol rate',default=2400)
parser.add_argument('--Fs', type=float, help='Sample rate, Fs/Rs must be an integer',default=9600)
parser.add_argument('--complex', dest="real", action='store_false', help='complex 2*int16 input samples (default real)')
parser.add_argument('-v', type=int, default=2, help='Verbose level (default 2)')
parser.add_argument('--plots', action='store_true', help='Enable plots when input data finished')
parser.set_defaults(real=True)
args = parser.parse_args()

if (args.fcentreHz < args.Rs/2) and args.real and (args.fcentreHz != 0):
   print("Warning - aliasing likely with real valued input samples, consider --complex")
modem = single_carrier(Rs=args.Rs, Fs=args.Fs, fcentreHz=args.fcentreHz)
assert modem.Npayload_syms == args.latent_dim

if args.real:
   int16s_per_sample = 1
else:
   int16s_per_sample = 2

if args.ber_test:
   tx_symbs = 1 - 2*(modem.rng.random(args.latent_dim) > 0.5) + 0*1j
   total_errors = 0
   total_bits = 0

if args.plots:
   rx_symb_log = np.array([], dtype=np.csingle)
   error_log = np.array([])
   norm_rx_timing_log = np.array([])
   phase_est_log = np.array([], dtype=np.csingle)

bytes_in = modem.nin*int16s_per_sample*struct.calcsize("h")
print(int16s_per_sample, modem.nin, bytes_in, file=sys.stderr)
while True:
   buffer = sys.stdin.buffer.read(bytes_in)
   if len(buffer) != bytes_in:
      break
   rx = np.zeros(modem.nin,dtype=np.csingle)
   if args.real:
      tmp = np.frombuffer(buffer,np.int16)
      print(len(tmp))
      rx.real = tmp
   else:
      tmp = np.frombuffer(buffer,np.int16)
      rx.real = tmp[::2]
      rx.imag = tmp[1::2]

   z_hat = modem.rx(rx)

   if modem.state == "sync":
      if args.ber_test:
         n_errors = np.sum(z_hat * tx_symbs < 0)
         total_errors += n_errors
         total_bits += len(tx_symbs)
      if args.plots:
         rx_symb_log = np.concatenate([rx_symb_log,modem.rx_symb_buf[args.latent_dim:]])
         error_log = np.append(error_log,n_errors)
         norm_rx_timing_log = np.append(norm_rx_timing_log, modem.norm_rx_timing)
         phase_est_log = np.append(phase_est_log, modem.phase_est_log)

   if args.v:
      print(f"state: {modem.state:6} nin: {modem.nin:4d} rx_timing: {modem.norm_rx_timing:5.2f} max_metric: {modem.max_Cs.real:5.2f}", end='', file=sys.stderr)
      print(f" fs_s: {modem.fs_s:4d} ph_ambig: {modem.phase_ambiguity:5.2f} g: {modem.g:5.2g}", end='', file=sys.stderr)
      if args.ber_test and modem.state == "sync":
         print(f" n_errors: {n_errors:4d}", end='', file=sys.stderr)
      print(file=sys.stderr)

   sys.stdout.buffer.write(z_hat) 
   bytes_in = modem.nin*int16s_per_sample*struct.calcsize("h")

# optional debug info
if args.ber_test:
   ber = 0
   if total_bits:
      ber = total_errors/total_bits
   print(f"total_bits: {total_bits:4d} total_errors: {total_errors:4d} BER: {ber:5.4f}", file=sys.stderr)

if args.plots:
   plt.figure(1)
   plt.plot(modem.g*rx_symb_log.real,modem.g*rx_symb_log.imag,'+'); plt.ylabel('Symbols')
   plt.axis([-2,2,-2,2])
   plt.figure(2)
   plt.subplot(211)
   plt.plot(error_log); plt.ylabel('Errors/frame')
   plt.subplot(212)
   plt.plot(norm_rx_timing_log,'+'); plt.ylabel('Fine Timing')
   plt.axis([0,len(norm_rx_timing_log),-0.5,0.5])
   plt.figure(3)
   plt.plot(np.angle(phase_est_log),'+'); plt.title('Phase Est')
   plt.axis([0,len(phase_est_log),-np.pi,np.pi])
   plt.figure(4)
   from matplotlib.mlab import psd
   P, frequencies = psd(rx,Fs=modem.Fs)
   PdB = 10*np.log10(P)
   plt.plot(frequencies,PdB); plt.title('PSD'); plt.grid()
   mx = 10*np.ceil(max(PdB)/10)
   plt.axis([-modem.Fs/2, modem.Fs/2, mx-40, mx])
   plt.show()
