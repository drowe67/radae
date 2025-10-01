"""
/*
  RADE V2 receiver: rate Fs complex samples in, features out.

  No pilots, ML fine timing and frame sync.

  Copyright (c) 2025 by David Rowe */

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

import os,sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf
from models_ft import ftDNNXcorr
from models_sync import FrameSyncNet

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('ft_model_name', type=str, help='path to fine timing model in .pth format')
parser.add_argument('sync_model_name', type=str, help='path to sync model in .pth format')
parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--bottleneck', type=int, default=3, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain (default 3)')
parser.add_argument('--cp', type=float, default=0.004, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0.04)')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--freq_offset', type=float, default=0, help='correct for this frequency offset')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling time offset in samples')
parser.add_argument('--correct_time_offset', type=int, default=-16, help='introduces a delay (or advance if -ve) in samples, applied in freq domain (default 0)')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--acq_time_target', type=float, default=1.0, help='Acquisition test mode mean acquisition time target (default 1.0)')
parser.add_argument('--stateful',  action='store_true', help='use stateful core decoder')
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation (fine timing)',default = 160,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension (fine timing)',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension (fine timing)',default = 160,required = False)
parser.add_argument('--write_Ry', type=str, default="", help='path to autocorrelation output feature file dim (seq_len,Ncp+M) .f32 format')
parser.add_argument('--write_delta_hat', type=str, default="", help='path to delta_hat output file dim (seq_len) .f32 format in .f32 format')
parser.add_argument('--write_delta_hat_rx', type=str, default="", help='path to delta_hat_rx file dim (seq_len) .f32 format in .f32 format')
parser.set_defaults(bpf=True)
parser.set_defaults(auxdata=True)
parser.add_argument('--pad_samples', type=int, default=0, help='Pad input with samples to simulate different timing offsets in rx signal')
parser.add_argument('--gain', type=float, default=1.0, help='manual gain control')
parser.add_argument('--agc', action='store_true', help='automatic gain control')
parser.add_argument('--w1_dec', type=int, default=96, help='Decoder GRU output dimension (default 96)')
parser.add_argument('--timing_onesec', action='store_true', help='first pass timing that just samples first 1s of sample')
parser.add_argument('--noframe_sync', action='store_true', help='disable frame sync (default enabled)')
parser.add_argument('--test_mode', action='store_true', help='inject test delta sequence')
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20
if args.auxdata:
    num_features += 1

# load RADE model
model = RADAE(num_features, latent_dim, EbNodB=100, Nzmf = 1,
              rate_Fs=True, bottleneck=args.bottleneck, cyclic_prefix=args.cp,
              time_offset=args.time_offset, correct_time_offset=args.correct_time_offset,
              stateful_decoder=args.stateful, w1_dec=args.w1_dec)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Load fine timing model
ft_nn = ftDNNXcorr(args.xcorr_dimension, args.gru_dim, args.output_dim)
ft_nn.load_state_dict(torch.load(args.ft_model_name,weights_only=True,map_location=torch.device('cpu')))
ft_nn.eval()

# Load sync model
sync_nn = FrameSyncNet(latent_dim)
sync_nn.load_state_dict(torch.load(args.sync_model_name,weights_only=True,map_location=torch.device('cpu')))
sync_nn.eval()

M = model.M
Ncp = model.Ncp
Ns = model.Ns           # number of rate Rs symbols per modem frame
Nmf = int(Ns*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc
w = model.w.cpu().detach().numpy()
Fs = float(model.Fs)

# load rx rate_Fs samples
rx = np.fromfile(args.rx, dtype=np.csingle)*args.gain
w_off = 2*np.pi*args.freq_offset/Fs
rx = rx*np.exp(-1j*w_off*np.arange(len(rx)))

# optional AGC
if args.agc:
   # target RMS level is PAPR ~ 3 dB less than peak of 1.0
   target = 1.0*10**(-3/20)
   gain = target/np.sqrt(np.mean(np.abs(rx)**2))
   print(f"AGC target {target:3.2f} gain: {gain:3.2e}")
   rx *= gain
# ensure an integer number of frames
rx = np.concatenate((np.zeros(args.pad_samples, dtype=np.complex64),rx))
#rx = rx[args.pad_samples:]
rx = rx[:Nmf*(len(rx)//Nmf)]
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)//Nmf}")

# TODO: fix contrast of spectrogram - it's not very useful
if args.plots:
   fig, ax = plt.subplots(2, 1,figsize=(6,12))
   ax[0].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[0].set_title('Before BPF')
   ax[0].axis([0,len(rx)/model.Fs,0,3000])

# BPF to remove some of the noise and improve acquisition
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
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

# Generate fine timing estimates - as a first pass for entire sample, as we don't
# have a stateful fine timing estimator

sequence_length = len(rx)//(Ncp+M) - 2
print(sequence_length)
Q = 8
Ry_norm = np.zeros((sequence_length+Q-1,Ncp+M),dtype=np.float32)
for s in np.arange(Q-1,sequence_length):
   for delta_hat in np.arange(Ncp+M):
      st = (s+1)*(Ncp+M) + delta_hat
      y_cp = rx[st-Ncp:st]
      y_m = rx[st-Ncp+M:st+M]
      Ry = np.dot(y_cp, np.conj(y_m))
      D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
      Ry_norm[s,delta_hat] = 2.*np.abs(Ry)/np.abs(D)
Ry_bar = np.zeros((sequence_length,Ncp+M),dtype=np.float32)
for s in np.arange(sequence_length):
   Ry_bar[s,:] = np.mean(Ry_norm[s:s+Q,:],axis=0)
if len(args.write_Ry):
   Ry_bar.flatten().tofile(args.write_Ry)
Ry_bar = torch.reshape(torch.tensor(Ry_bar),(1,Ry_bar.shape[0],Ry_bar.shape[1]))

logits_softmax = ft_nn(Ry_bar)
delta_hat = torch.argmax(logits_softmax, 2).cpu().detach().numpy().flatten().astype('float32')
if len(args.write_delta_hat):
   delta_hat.flatten().tofile(args.write_delta_hat)

# concat rx vector with zeros at either end so we can extract an integer number of symbols
# despite fine timing offset
len_rx = len(rx)
#rx = np.concatenate((np.zeros(Ncp+M,dtype=np.complex64),rx,np.zeros(Ncp+M,dtype=np.complex64)))
rx = np.concatenate((rx,np.zeros(Ncp+M,dtype=np.complex64)))
# extract a vector corrected for fine timing est
if args.timing_onesec:
   # Use average of first 1 second of FT est to obtain ideal sampling point, avoid
   # first few symbols as they appear to be start up transients.  Really basic first pass.
   # Note conversion in time reference, delta_hat is referenced to Ncp/M boundary, but
   # receiver uses start of CP.
   delta_hat_rx = int(np.mean(delta_hat[10:50])) - Ncp
   print(f"sampling instant: {delta_hat_rx:d}")
   rx = rx[Ncp+M+delta_hat_rx:Ncp+M+delta_hat_rx+len_rx]
   # obtain z_hat from OFDM rx signal
   rx = torch.tensor(rx, dtype=torch.complex64)
   z_hat = model.receiver(rx,run_decoder=False)
   print("z_hat.shape",z_hat.shape)
else:
   # use timing estimates as they evolve
   Nframes = sequence_length//model.Ns
   z_hat = torch.zeros((1,Nframes, model.latent_dim), dtype=torch.float32)
   delta_hat_rx = np.zeros(Nframes,dtype=np.int16)
   test_delta_hat_rx = 2
   # note only one time estimate per frame (Ns symbols), we don't want a timing change
   # mid frame
   for i in np.arange(0,Nframes):
      if args.test_mode:
         delta_hat_rx[i] = int(np.mean(delta_hat[10:50])) - Ncp
         if i == 50:
            test_delta_hat_rx -= 3
         if i == 60:
            test_delta_hat_rx += 3
         delta_hat_rx[i] = test_delta_hat_rx
      else:
         delta_hat_rx[i] = int(delta_hat[model.Ns*i]-Ncp)

      st = (model.Ns*i)*(Ncp+M) + delta_hat_rx[i]
      st = max(st,0)
      en = st + model.Ns*(Ncp+M)
      if i < 10:
         print(i,delta_hat_rx[i],st,en)
      # extract rx samples for i-th frame
      rx_i = torch.tensor(rx[st:en], dtype=torch.complex64)
      #print(rx_i.shape)
      # run receiver to extract i-th freq domain OFDM symbols z_hat
      az_hat = model.receiver(rx_i,run_decoder=False)
      #print(z_hat.shape, z_hat[0,i,:].shape,az_hat.shape,az_hat[0,:,:].shape )
      z_hat[0,i,:] = az_hat
   print("z_hat.shape",z_hat.shape)
print(delta_hat_rx)

if len(args.write_delta_hat_rx):
   np.float32(delta_hat_rx).flatten().tofile(args.write_delta_hat_rx)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)
      
# now perform ML frame sync, two possibilities offset by one OFDM symbol (half a z vector)
if args.noframe_sync == False:
   Nsync_syms = 10 # average sync metric over this many OFDM symbols
   print(z_hat.shape)
   z_hat = torch.reshape(z_hat,(1,-1,latent_dim//2))
   print(z_hat.shape)
   sync_st=10
   sync_even = torch.mean(sync_nn(torch.reshape(z_hat[0,sync_st:sync_st+Nsync_syms,:],(1,-1,latent_dim))))
   sync_odd = torch.mean(sync_nn(torch.reshape(z_hat[0,sync_st+1:sync_st+1+Nsync_syms,:],(1,-1,latent_dim))))
   print(f"sync_even: {sync_even:5.2f} sync_odd: {sync_odd:5.2f}")
   if sync_even > sync_odd:
      offset = 0
   else:
      offset = 1
   z_hat_len = z_hat.shape[1]
   z_hat = z_hat[:,offset:z_hat_len-offset,:]
   z_hat = torch.reshape(z_hat,(1,-1,latent_dim))

# run RADE decoder
features_hat = model.core_decoder(z_hat)
features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:nb_total_features-num_features]], dim=-1)
#print(features_hat.shape)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)
