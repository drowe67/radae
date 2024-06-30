"""

  Radio Autoencoder streaming receiver: 
  
  rate Fs complex float samples in, features out.
  rate Fs real int16 samples in, features out.

  Designed to connected to a SDR to perform real time RADAE decoding on 
  received sample streams.  Full function state machine and continous 
  updates of timing, freq offsets and amplituide estimates.
  
  Copyright (c) 2024 by David Rowe

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

import os, sys, argparse, struct
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf,acquisition

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('rxfile', type=argparse.FileType("rb"), default=sys.stdin, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format (default stdin)')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--ber_test', type=str, default="", help='symbols are PSK bits, compare to z.f32 file to calculate BER')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--bottleneck', type=int, default=3, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--write_Dt', type=str, default="", help='Write D(t,f) matrix on last modem frame')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--fmax_target', type=float, default=0.0, help='Acquisition test mode freq offset target (default 0.0)')
parser.set_defaults(bpf=True)
args = parser.parse_args()

# handle use of stdin
if hasattr(args.rxfile, "buffer"):
   args.samplefile = args.samplefile.buffer

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, 
              pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
              coarse_mag=True,time_offset=-16, bottleneck=args.bottleneck)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

M = model.M
Ncp = model.Ncp
Ns = model.Ns               # number of data symbols between pilots
Nmf = int((Ns+1)*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc

# TODO: we need a streaming BPF with state

"""
# load rx rate_Fs samples, BPF to remove some of the noise and improve acquisition
rx = np.fromfile(args.rx, dtype=np.csingle)
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)/Nmf}")

w = model.w.cpu().detach().numpy()
Ntap = 0
if args.bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   rx = complex_bpf(Ntap, model.Fs, bandwidth,centre, rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[1].axis([0,len(rx)/model.Fs,0,3000])
   ax[1].set_title('After BPF')
"""

p = np.array(model.p) 
frange = 100                                # coarse grid -frange/2 ... + frange/2
fstep = 2.5                                 # coarse grid spacing in Hz
Fs = model.Fs
Rs = model.Rs

acq = acquisition(Fs,Rs,M,Nmf,p)

"""
# optional acq_test variables 
tmax_candidate_target = Ncp + Ntap/2
acq_pass = 0
acq_fail = 0
"""

tmax_candidate = 0 
acquired = False
state = "search"
mf = 1

fmt="<ff"
rx = np.zeros(2*Nmf+M,np.csingle)
rx_buf = np.zeros(Nmf,np.csingle)
decode_buf = np.zeros(0,np.csingle)
nbuf = 0
rx_phase = 1 + 1j*0
rx_phase_vec = np.zeros(Nmf,np.csingle)

while True:
   buffer = args.rxfile.read(struct.calcsize(fmt))
   if not buffer:
      break
   sampleIQ = struct.unpack(fmt, buffer)
   rx_buf[nbuf] = sampleIQ[0] + 1j*sampleIQ[1]
   nbuf += 1
   if nbuf == Nmf:
      rx[:Nmf+M] = rx[Nmf:]
      rx[Nmf+M:] = rx_buf
      
      if state == "search" or state == "candidate":
         candidate, tmax, fmax = acq.detect_pilots(rx)
   
      # print current state
      print(f"{mf:2d} state: {state:10s} Dthresh: {acq.Dthresh:5.2f} Dtmax12: {acq.Dtmax12:5.2f} tmax: {tmax:4d} tmax_candidate: {tmax_candidate:4d} fmax: {fmax:6.2f}")

      # iterate state machine  
      next_state = state
      if state == "search":
         if candidate:
            next_state = "candidate"
            tmax_candidate = tmax
            valid_count = 1
      elif state == "candidate":
         # look for 3 consecutive matches with about the same timing offset  
         if candidate and np.abs(tmax-tmax_candidate) < 0.02*M:
            valid_count = valid_count + 1
            if valid_count > 3:
               next_state = "sync"
               acquired = True
               ffine_range = np.arange(fmax-10,fmax+10,0.25)
               fmax = acq.refine(rx, tmax, fmax, ffine_range)
               w = 2*np.pi*fmax/Fs
         else:
            next_state = "search"
      elif state == "sync":
         for n in range(Nmf):
            rx_phase = rx_phase*np.exp(-1j*w)
            rx_phase_vec[n] = rx_phase
         decode_buf = np.append(decode_buf,rx[tmax-Ncp:tmax-Ncp+Nmf]*rx_phase_vec)
      state = next_state
      nbuf = 0           
      mf += 1


if not acquired:
   print("Acquisition failed....")
   quit()

# run vanilla (non streaming) decoder for now 
# TODO: replace with stateful/streaming decoder
rx = torch.tensor(decode_buf, dtype=torch.complex64)
model.to(device)
rx = rx.to(device)
features_hat, z_hat = model.receiver(rx)

z_hat = z_hat.cpu().detach().numpy().flatten().astype('float32')

# BER test useful for calibrating link.  To measure BER we compare the received symnbols 
# to the known transmitted symbols.  However due to acquisition delays we may have lost several
# modem frames in the received sequence.
if len(args.ber_test):
   # every time acq shifted Nmf (one modem frame of samples), we shifted this many latents:
   num_latents_per_modem_frame = model.Nzmf*model.latent_dim
   #print(num_latents_per_modem_frame)
   z = np.fromfile(args.ber_test, dtype=np.float32)
   #print(z.shape, z_hat.shape)
   best_BER = 1
   # to find best alignment look for lowerest BER over a range of shifts
   for f in np.arange(20):
      n_syms = min(len(z),len(z_hat))
      n_errors = np.sum(-z[:n_syms]*z_hat[:n_syms]>0)
      n_bits = len(z)
      BER = n_errors/n_bits
      if BER < best_BER:
         best_BER = BER
         print(f"f: {f:2d} n_bits: {n_bits:d} n_errors: {n_errors:d} BER: {BER:5.3f}")
      z = z[num_latents_per_modem_frame:]

features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

# write real valued latent vectors
if len(args.write_latent):
   z_hat.tofile(args.write_latent)
