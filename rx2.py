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
parser.add_argument('--freq_offset', type=float, help='insert a frequency offset, e.g. for manual coarse freq estimation')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling time offset in samples')
parser.add_argument('--correct_time_offset', type=int, default=-16, help='introduces a delay (or advance if -ve) in samples, applied in freq domain (default 0)')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--acq_time_target', type=float, default=1.0, help='Acquisition test mode mean acquisition time target (default 1.0)')
parser.add_argument('--stateful',  action='store_true', help='use stateful core decoder')
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation (fine timing)',default = 160,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension (fine timing)',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension (fine timing)',default = 160,required = False)
parser.add_argument('--write_Ry', type=str, default="", help='path to autocorrelation output feature file dim (Ncp+M) .f32 format in .f32 format')
parser.set_defaults(bpf=True)
parser.set_defaults(auxdata=True)
parser.add_argument('--pad_samples', type=int, default=0, help='Pad input with samples to simulate different timing offsets in rx signal')
parser.add_argument('--gain', type=float, default=1.0, help='manual gain control')
parser.add_argument('--w1_dec', type=int, default=96, help='Decoder GRU output dimension (default 96)')
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

# load rx rate_Fs samples, BPF to remove some of the noise and improve acquisition
rx = np.fromfile(args.rx, dtype=np.csingle)*args.gain
# ensure an integer number of frames
rx = np.concatenate((np.zeros(args.pad_samples, dtype=np.complex64),rx))
rx = rx[:Nmf*(len(rx)//Nmf)]
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)//Nmf}")

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
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

# Generate fine timing estimates - as a first pass for entire sample, as we don't
# have a stateful fine tinming estimator

sequence_length = len(rx)//(Ncp+M) - 2
Q = 8
Ry = np.zeros((sequence_length+Q-1,Ncp+M),dtype=np.float32)
for s in np.arange(Q-1,sequence_length):
   for delta_hat in np.arange(Ncp+M):
      st = (s+1)*(Ncp+M) + delta_hat
      y1 = rx[st-M:st-M+Ncp]
      y2 = rx[st:st+Ncp]
      num = np.abs(np.dot(y1, np.conj(y2)))
      den = np.abs((np.dot(y1, np.conj(y1)) + np.dot(y2, np.conj(y2))))
      Ry[s,delta_hat] = num/den
Ry_smooth = np.zeros((sequence_length,Ncp+M),dtype=np.float32)
for s in np.arange(sequence_length):
   Ry_smooth[s,:] = np.mean(Ry[s:s+Q,:],axis=0)
print(Ry_smooth.shape)
if len(args.write_Ry):
   Ry_smooth.flatten().tofile(args.write_Ry)
Ry_smooth = torch.reshape(torch.tensor(Ry_smooth),(1,Ry_smooth.shape[0],Ry_smooth.shape[1]))

logits_softmax = ft_nn(Ry_smooth)
delta_hat = torch.argmax(logits_softmax, 2).cpu().detach().numpy().flatten().astype('float32')

# Use average of first 1 second of FT est to obtain ideal sampling point, avoid
# first few symbols as they appear to be start up transients. This is the location
# of the first sample after the CP (see hf3 doc figure)
s = round(np.mean(delta_hat[10:50]-M))
print(f"sampling instant: {s:d}")
len_rx = len(rx)
# concat rx vector with zeros at either end so we can extract an integer number of symbols
# despite fine timing offset
rx = np.concatenate((np.zeros(Ncp+M,dtype=np.complex64),rx,np.zeros(Ncp+M,dtype=np.complex64)))
# extract a vector corrected for fine timing est
rx = rx[Ncp+M+s:Ncp+M+s+len_rx]
# dummy frame sync
#rx = rx[(Ncp+M):]
#sys.exit(1)

# obtain z_hat from OFDM rx signal
rx = torch.tensor(rx, dtype=torch.complex64)
z_hat = model.receiver(rx,run_decoder=False)

# now perform ML frame sync, two possibilities offset by one OFDM symbol (half a z vector)
Nsync_syms = 10 # average sync metric over this many OFDM symbols
z_hat = torch.reshape(z_hat,(1,-1,latent_dim//2))
sync_even = torch.mean(sync_nn(torch.reshape(z_hat[0,:Nsync_syms,:],(1,-1,latent_dim))))
sync_odd = torch.mean(sync_nn(torch.reshape(z_hat[0,1:Nsync_syms+1,:],(1,-1,latent_dim))))
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
print(features_hat.shape)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)
