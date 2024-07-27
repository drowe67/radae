"""
/*
  Radio Autoencoder receiver: rate Fs complex samples in, features out.

  Bare bones acquisition that find a valid modem frame, and decodes the
  entire sample using a fixed timing and freq offset estimate.  Works
  OK for 10 second samples, tested on many HF channels around the world
  in April 2024 OTA test campaign.

  Copyright (c) 2024 by David Rowe */

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
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf,acquisition,receiver_one

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--pilots', action='store_true', help='insert pilot symbols')
parser.add_argument('--ber_test', type=str, default="", help='symbols are PSK bits, compare to z.f32 file to calculate BER')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--pilot_eq', action='store_true', help='use pilots to EQ data symbols using classical DSP')
parser.add_argument('--freq_offset', type=float, help='manually specify frequency offset')
parser.add_argument('--cp', type=float, default=0.0, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0)')
parser.add_argument('--coarse_mag', action='store_true', help='Coarse magnitude correction (fixes --gain)')
parser.add_argument('--time_offset', type=int, default=0, help='sampling time offset in samples')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--bottleneck', type=int, default=1, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--write_Dt', type=str, default="", help='Write D(t,f) matrix on last modem frame')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--fmax_target', type=float, default=0.0, help='Acquisition test mode freq offset target (default 0.0)')
parser.add_argument('--acq_time_target', type=float, default=1.0, help='Acquisition test mode mean acquisition time target (default 1.0)')
parser.add_argument('--rx_one',  action='store_true', help='Use single frame receiver')
parser.add_argument('--stateful',  action='store_true', help='use stateful core decoder')
parser.set_defaults(bpf=True)
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, 
              pilots=args.pilots, pilot_eq=args.pilot_eq, eq_mean6 = False, cyclic_prefix=args.cp,
              coarse_mag=args.coarse_mag,time_offset=args.time_offset, bottleneck=args.bottleneck,
              stateful_decoder=args.stateful)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.core_decoder_statefull_load_state_dict()

M = model.M
Ncp = model.Ncp
Ns = model.Ns               # number of data symbols between pilots
Nmf = int((Ns+1)*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc
w = model.w.cpu().detach().numpy()

# load rx rate_Fs samples, BPF to remove some of the noise and improve acquisition
rx = np.fromfile(args.rx, dtype=np.csingle)
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)/Nmf}")

# TODO: fix contrast of spectrogram - it's not very useful
if args.plots:
   fig, ax = plt.subplots(2, 1,figsize=(6,12))
   ax[0].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[0].set_title('Before BPF')
   ax[0].axis([0,len(rx)/model.Fs,0,3000])

Ntap = 0
if args.bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   bpf = complex_bpf(Ntap, model.Fs, bandwidth,centre)
   rx = bpf.bpf(rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[1].axis([0,len(rx)/model.Fs,0,3000])
   ax[1].set_title('After BPF')


# Acquisition - 1 sample resolution timing, coarse/fine freq offset estimation

if args.pilots:
   p = np.array(model.p) 
   frange = 100                                # coarse grid -frange/2 ... + frange/2
   fstep = 2.5                                 # coarse grid spacing in Hz
   Fs = model.Fs
   Rs = model.Rs

   acq = acquisition(Fs,Rs,M,Ncp,Nmf,p,model.pend)
 
   # optional acq_test variables 
   tmax_candidate_target = Ncp + Ntap/2
   acq_pass = 0
   acq_fail = 0

   tmax_candidate = 0 
   acquired = False
   state = "search"
   mf = 1
   
   if len(args.write_Dt):
      fD=open(args.write_Dt,'wb')

   while not acquired and len(rx) >= 2*Nmf+M:
      candidate, tmax, fmax = acq.detect_pilots(rx[:2*Nmf+M+Ncp])
      if len(args.write_Dt):
         acq.Dt1.tofile(fD)
      
      # post process with a state machine that looks for 3 consecutive matches with about the same timing offset      
      print(f"{mf:2d} state: {state:10s} Dthresh: {acq.Dthresh:8.2f} Dtmax12: {acq.Dtmax12:8.2f} tmax: {tmax:4d} tmax_candidate: {tmax_candidate:4d} fmax: {fmax:6.2f}")

      next_state = state
      if state == "search":
         if candidate:
            next_state = "candidate"
            tmax_candidate = tmax
            valid_count = 1
      elif state == "candidate":
         if candidate and np.abs(tmax-tmax_candidate) < 0.02*M:
            valid_count = valid_count + 1
            if valid_count > 3:
               if args.acq_test:
                  next_state = "search"
                  ffine_range = np.arange(fmax-10,fmax+10,0.25)
                  tfine_range = np.arange(tmax-1,tmax+2)
                  tmax,fmax = acq.refine(rx, tmax, fmax, tfine_range, ffine_range)
                  # allow 2ms spread in timing (MPP channel extremes) and +/- 5 Hz in freq, which fine freq can take care of
                  coarse_timing_ok = np.abs(tmax - tmax_candidate_target) < 0.0025*Fs
                  coarse_freq_ok = np.abs(fmax - args.fmax_target) <= 5.0
                  print(f"Acquired! Timing: {coarse_timing_ok:d} Freq: {coarse_freq_ok:d} ",end='')
                  if coarse_timing_ok and coarse_freq_ok:
                     acq_pass = acq_pass + 1
                     print("")
                  else:
                     acq_fail = acq_fail + 1
                     print(f"fmax: {fmax:6.2f} targ: {args.fmax_target:6.2f}")
               else:
                  print("Acquired!")
                  acquired = True
         else:
            next_state = "search"
      state = next_state
                  
      # advance one frame and repeat
      rx = rx[Nmf:-1]
      mf += 1

   if args.acq_test:
      mean_acq_time = (Nmf*mf/Fs)/acq_pass
      pfail = acq_fail/(acq_pass+acq_fail)
      print(f"Acq Test Passes: {acq_pass:d} Fails: {acq_fail:d} Pfail: {pfail:5.2f} Mean Acq time: {mean_acq_time:5.2f} s")
      if (pfail < 0.2) and (mean_acq_time < args.acq_time_target):
         print("PASS")

   if not acquired:
      print("Acquisition failed....")
      quit()

   # frequency refinement, use two sets of pilots
   ffine_range = np.arange(fmax-10,fmax+10,0.25)
   tfine_range = np.arange(tmax-1,tmax+2)
   
   tmax,fmax = acq.refine(rx, tmax, fmax, tfine_range, ffine_range)
   print(f"refined fmax: {fmax:f}")

   if args.plots:
      fig, ax = plt.subplots(2, 1,figsize=(6,12))
      ax[0].set_title('Dt complex plane')
      ax[0].plot(acq.Dt1[:,acq.f_ind_max].real, acq.Dt1[:,acq.f_ind_max].imag,'b+')
      circle1 = plt.Circle((0,0), radius=acq.Dthresh, color='r')
      ax[0].add_patch(circle1)
      ax[1].hist(np.abs(acq.Dt1[:,acq.f_ind_max]))
      ax[1].set_title('|Dt| histogram')

      fig1, ax1 = plt.subplots(2, 1,figsize=(6,12))
      ax1[0].plot(acq.fcoarse_range, np.abs(acq.Dt1[tmax,:]),'b+')
      ax1[0].set_title('|Dt| against f (coarse)')
      ax1[1].plot(ffine_range, np.abs(acq.D_fine),'b+')
      ax1[1].set_title('|Dt| against f (fine)')
   
   rx = rx[tmax-Ncp:]
   if args.freq_offset is not None:
      fmax = args.freq_offset
      print(fmax)
   w = 2*np.pi*fmax/Fs
   rx = rx*np.exp(-1j*w*np.arange(len(rx)))

# push model to device and run receiver
rx = torch.tensor(rx, dtype=torch.complex64)
model.to(device)
rx = rx.to(device)

if args.rx_one:
   receiver = receiver_one(model.latent_dim,model.Fs,model.M,model.Ncp,model.Wfwd,
                           model.Nc,model.Ns,model.w,model.P,model.bottleneck,
                           model.pilot_gain,model.time_offset,model.coarse_mag)
   Nmodem_frames = (len(rx)-(M+Ncp))//Nmf
   features_hat = torch.empty(1,0,model.feature_dim)
   z_hat = torch.empty(1,0,model.latent_dim)
   #print(Nmodem_frames,features_hat.shape)

   for f in range(Nmodem_frames):
      z_hat1 = receiver.receiver_one(rx[f*Nmf:(f+1)*Nmf+M+Ncp])
      #print(z_hat1.shape)
      assert(z_hat1.shape[1] == model.Nzmf)
      for i in range(model.Nzmf):
         features_hat = torch.cat([features_hat, model.core_decoder_statefull(z_hat1[:,i:i+1,:])],dim=1)
      z_hat = torch.cat([z_hat, z_hat1],dim=1)
      #print(f,features_hat.shape,z_hat.shape)
else:
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
   #errors = torch.sign(-z*z_hat) > 0
   #errors = torch.reshape(errors,(-1,latent_dim))
   #print(errors.shape)
   #print(torch.sum(errors,dim=1))

features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

# write real valued latent vectors
if len(args.write_latent):
   z_hat.tofile(args.write_latent)

if args.plots:
   plt.figure(4)
   plt.plot(z_hat[0:-2:2], z_hat[1:-1:2],'+')
   plt.title('Scatter')
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

