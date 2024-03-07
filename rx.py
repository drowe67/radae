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
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, pilots=args.pilots, pilot_eq=args.pilot_eq)
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
   
   return np.convolve(x,h)

# load rx rate_Fs samples, BPF to remove some of the noise and improve acquisition
rx = np.fromfile(args.rx, dtype=np.csingle)

# TODO: fix contrast of spectrogram - it's not very useful
if args.plots:
   fig, ax = plt.subplots(2, 1,figsize=(6,12))
   fig.suptitle('Rx before and after BPF')
   ax[0].specgram(rx,NFFT=256,Fs=model.get_Fs())
   ax[0].axis([0,len(rx)/model.get_Fs(),0,2000])
   ax[0].title
rx = complex_bpf(8000,1200,900,rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.get_Fs())
   ax[1].axis([0,len(rx)/model.get_Fs(),0,2000])
 
# acquisition - coarse & fine timing

if args.pilots:
   M = int(model.get_Fs()/model.get_Rs())
   Ns = (model.get_Ns()+1)
   Nmf = int(Ns*M)                             # number of samples in one modem frame
   p = model.p                                 # pilot sequence
   D = np.zeros(Nmf, dtype=np.csingle)         # correlation at various time offsets
   Dtmax = 0
   tmax = 0
   Pacq_error = 0.001
   acquired = False
   while not acquired and len(rx) >= Nmf+M:
      # search modem frame for maxima
      for t in range(Nmf):
         D[t] = np.dot(np.conj(rx[t:t+model.M]),p)
         if np.abs(D[t]) > Dtmax:
            Dtmax = np.abs(D[t])
            tmax = t
      
      sigma_est = np.std(D)
      Dthresh = sigma_est*np.sqrt(-np.log(Pacq_error))
      print(f"sigma: {sigma_est:f} Dthresh: {Dthresh:f} Dtmax: {Dtmax:f} tmax: {tmax:d}")
      if Dtmax > Dthresh:
         acquired = True
         print("Acquired!")
      else:
         # advance one frame and search again
         rx = rx[Nmf:-1]
   if not acquired:
      print("Acquisition failed....")
      quit()
   if args.plots:
      fig, ax = plt.subplots(2, 1,figsize=(6,12))
      fig.suptitle('Dt complex plane and |Dt| histogram')
      ax[0].plot(D.real, D.imag,'b+')
      circle1 = plt.Circle((0,0), radius=Dthresh, color='r')
      ax[0].add_patch(circle1)
      ax[1].hist(np.abs(D))
 
   print(len(rx))
   rx = rx[tmax:]
   print(len(rx))

   # magnitude normalisation

   r = rx[:M]
   g = np.dot(np.conj(r),r)/np.dot(np.conj(p),p)
   print(f"g: {g:f}")
   #rx = rx/g
   #quit()

# push model to device and run receiver
rx = torch.tensor(rx, dtype=torch.complex64)
model.to(device)
rx = rx.to(device)
features_hat, z_hat = model.receiver(rx)

z_hat = z_hat.cpu().detach().numpy().flatten().astype('float32')

# BER test useful for calibrating link
if len(args.ber_test):
   z = torch.tensor(np.fromfile(args.ber_test, dtype=np.float32))
   print(z.shape, z_hat.shape)
   n_errors = torch.sum(-z*z_hat>0)
   n_bits = torch.numel(z)
   BER = n_errors/n_bits
   print(f"n_bits: {n_bits:d} BER: {BER:5.3f}")
   errors = torch.sign(-z*z_hat) > 0
   errors = torch.reshape(errors,(-1,latent_dim))
   print(errors.shape)
   print(torch.sum(errors,dim=1))

features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

# write real valued latent vectors
if len(args.write_latent):
   z_hat = z_hat.cpu().detach().numpy().flatten().astype('float32')
   z_hat.tofile(args.write_latent)

if args.plots:
   plt.figure(3)
   plt.plot(z_hat[0:-2:2], z_hat[1:-1:2],'+')
   plt.title('Scatter')
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

