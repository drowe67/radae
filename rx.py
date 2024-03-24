"""
/*
  Radio Autoencoder receiver: rate Fs complex samples in, features out.

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
from radae import RADAE

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
              coarse_mag=args.coarse_mag,time_offset=args.time_offset)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

def complex_bpf(Fs_Hz, bandwidth_Hz, centre_freq_Hz, x):
   B = bandwidth_Hz/Fs_Hz
   alpha = 2*np.pi*centre_freq_Hz/Fs_Hz
   Ntap=101
   h = np.zeros(Ntap, dtype=np.csingle)

   for i in range(Ntap):
      n = i-(Ntap-1)/2
      h[i] = B*np.sinc(n*B)
   
   x_baseband = x*np.exp(-1j*alpha*np.arange(len(x)))
   x_filt = np.convolve(x_baseband,h)
   return x_filt*np.exp(1j*alpha*np.arange(len(x_filt)))

M = model.M
Ncp = model.Ncp
Ns = model.Ns               # number of data symbols between pilots
Nmf = int((Ns+1)*(M+Ncp))   # number of samples in one modem frame

# load rx rate_Fs samples, BPF to remove some of the noise and improve acquisition
rx = np.fromfile(args.rx, dtype=np.csingle)
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx) // Nmf}")

# TODO: fix contrast of spectrogram - it's not very useful
if args.plots:
   fig, ax = plt.subplots(2, 1,figsize=(6,12))
   ax[0].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[0].set_title('Before BPF')
   ax[0].axis([0,len(rx)/model.Fs,0,3000])

Nc = model.Nc
w = model.w.cpu().detach().numpy()
bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
rx = complex_bpf(model.Fs,bandwidth,centre,rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[1].axis([0,len(rx)/model.Fs,0,3000])
   ax[1].set_title('After BPF')

# Acquisition - 1 sample resolution timing, coarse/fine freq offset estimation

if args.pilots:
   p = np.array(model.p)                       # pilot sequence
   frange = 100                                # coarse grid -frange/2 ... + frange/2
   fstep = 5                                   # coarse grid spacing in Hz
   Fs = model.Fs
   Rs = model.Rs

   # correlation at various time and freq offsets
   fcoarse_range = np.arange(-frange/2,frange/2,fstep)
   D = np.zeros((Nmf,len(fcoarse_range)), dtype=np.csingle)

   tmax_candidate = 0 
   Pacq_error = 0.0001
   acquired = False
   state = "search"

   # pre-calculate to speeds things up a bit
   p_w = np.zeros((len(fcoarse_range), M), dtype=np.csingle)
   f_ind = 0
   for f in fcoarse_range:
      w = 2*np.pi*f/Fs
      p_w[f_ind,] = np.exp(1j*w*np.arange(M)) * p

      f_ind + f_ind + 1

   while not acquired and len(rx) >= Nmf+M:
      # Search modem frame for maxima in correlation between pilots and received signal, over
      # a grid of time and frequency steps.  Note we only correlate on the M samples after the
      # cyclic prefix, so tmax will be Ncp samples after the start of the modem frame
      Dtmax = 0
      tmax = 0
      fmax = 0

      for t in range(Nmf):
         f_ind = 0
         for f in fcoarse_range:
            #D[t,f_ind] = np.dot(np.conj(rx[t:t+M]),p_w[f_ind,:])
            w = 2*np.pi*f/Fs
            w_vec = np.exp(-1j*w*np.arange(M))
            D[t,f_ind] = np.dot(np.conj(w_vec*rx[t:t+M]),p)

            if np.abs(D[t,f_ind]) > Dtmax:
               Dtmax = np.abs(D[t,f_ind])
               tmax = t
               fmax = f 
               f_ind_max =  f_ind
            f_ind = f_ind + 1
      
      # Ref: freedv_low.pdf "Coarse Frequency Estimation"
      sigma_est = np.std(D)
      Dthresh = sigma_est*np.sqrt(-np.log(Pacq_error))

      candidate = False
      if Dtmax > Dthresh:
         candidate = True

      # post process with a state machine that looks for 3 consecutive matches with about the same tmining offset      
      if candidate:
         print(f"state: {state:10s} Dthresh: {Dthresh:f} Dtmax: {Dtmax:f} tmax: {tmax:4d} tmax_candidate: {tmax_candidate:4d} fmax: {fmax:f}")

      next_state = state
      match state:
         case "search":
            if candidate:
               next_state = "candidate"
               tmax_candidate = tmax
               valid_count = 1
         case "candidate":
            if candidate and np.abs(tmax-tmax_candidate) < 0.02*M:
               valid_count = valid_count + 1
               if valid_count > 3:
                  acquired = True
                  print("Acquired!")
            else:
               next_state = "search"
      state = next_state
                  
      # advance one frame and repeat
      rx = rx[Nmf:-1]

   if not acquired:
      print("Acquisition failed....")
      quit()

   # frequency refinement, use two sets of pilots
   ffine_range = np.arange(fmax-5,fmax+5,1)
   #print(ffine_range)
   D_fine = np.zeros(len(ffine_range), dtype=np.csingle)
   f_ind = 0
   fmax_fine = fmax
   for f in ffine_range:
      w = 2*np.pi*f/Fs
      # current pilot samples at start of this modem frame
      w_vec = np.exp(-1j*w*np.arange(M))
      D_fine[f_ind] = np.dot(np.conj(w_vec*rx[tmax:tmax+M]),p)
      # next pilot samples at end of this modem frame
      w_vec = np.exp(-1j*w*(Nmf+np.arange(M)))
      D_fine[f_ind] = D_fine[f_ind] + np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),p)

      if np.abs(D_fine[f_ind]) > Dtmax:
         Dtmax = np.abs(D_fine[f_ind])
         fmax = f 
      f_ind = f_ind + 1
   print(f"refined fmax: {fmax:f}")

   if args.plots:
      fig, ax = plt.subplots(2, 1,figsize=(6,12))
      ax[0].set_title('Dt complex plane')
      ax[0].plot(D[:,f_ind_max].real, D[:,f_ind_max].imag,'b+')
      circle1 = plt.Circle((0,0), radius=Dthresh, color='r')
      ax[0].add_patch(circle1)
      ax[1].hist(np.abs(D[:,f_ind_max]))
      ax[1].set_title('|Dt| histogram')

      fig1, ax1 = plt.subplots(2, 1,figsize=(6,12))
      ax1[0].plot(fcoarse_range, np.abs(D[tmax,:]),'b+')
      ax1[0].set_title('|Dt| against f (coarse)')
      ax1[1].plot(ffine_range, np.abs(D_fine),'b+')
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
features_hat, z_hat = model.receiver(rx)

z_hat = z_hat.cpu().detach().numpy().flatten().astype('float32')

# BER test useful for calibrating link.  To mneasure BER we compare the received symnbols 
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

