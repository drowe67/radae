"""
/*
  RADE V2 receiver: rate Fs complex samples in, features out.

  No pilots, DSP acquisition, ML frame sync.

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
from models_sync import FrameSyncNet

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('frame_sync_model_name', type=str, help='path to frame sync model in .pth format')
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
parser.add_argument('--write_Ry_norm', type=str, default="", help='path to normalised autocorrelation output feature file dim (seq_len,Ncp+M) .f32 format')
parser.add_argument('--write_Ry_smooth', type=str, default="", help='path to smoothed autocorrelation output feature file dim (seq_len,Ncp+M) .f32 format')
parser.add_argument('--write_delta_hat', type=str, default="", help='path to delta_hat output file dim (seq_len) in .int16 format')
parser.add_argument('--write_delta_hat_pp', type=str, default="", help='path to delta_hat_pp output file dim (seq_len) in .int16 format')
parser.add_argument('--write_Ry_max', type=str, default="", help='path to Ty_max output file dim (seq_len) in .f32 format')
parser.add_argument('--write_sig_det', type=str, default="", help='path to signal detection flag output file dim (seq_len) in .int16 format')
parser.add_argument('--write_freq_offset', type=str, default="", help='path to freq offset est output file dim (seq_len) in .float32 format')
parser.add_argument('--write_freq_offset_smooth', type=str, default="", help='path to smoothed freq offset est output file dim (seq_len) in .float32 format')
parser.add_argument('--write_delta_hat_rx', type=str, default="", help='path to delta_hat_rx file dim (seq_len) in .f32 format')
parser.add_argument('--write_state', type=str, default="", help='path to sync state machine output file dim (seq_len) in .int16 format')
parser.add_argument('--write_frame_sync', type=str, default="", help='path to frame sync output file dim (seq_len,2) in .int16 format')
parser.add_argument('--read_delta_hat', type=str, default="", help='path to delta_hat input file dim (seq_len) in .f32 format')
parser.add_argument('--fix_delta_hat', type=int,  default=0, help='disable timing estimation and used fixed delta_hat (default: use timing estimation)')
parser.set_defaults(bpf=True)
parser.set_defaults(auxdata=True)
parser.set_defaults(verbose=True)
parser.add_argument('--pad_samples', type=int, default=0, help='Pad input with samples to simulate different timing offsets in rx signal')
parser.add_argument('--gain', type=float, default=1.0, help='manual gain control')
parser.add_argument('--agc', action='store_true', help='automatic gain control')
parser.add_argument('--w1_dec', type=int, default=96, help='Decoder GRU output dimension (default 96)')
parser.add_argument('--nofreq_offset', action='store_true', help='disable freq offset correction (default enabled)')
parser.add_argument('--test_mode', action='store_true', help='inject test delta sequence')
parser.add_argument('--hangover', type=int, default=75, help='Number of symbols of no signal before returning to noise state (default 75)')
parser.add_argument('--quiet', action='store_false', dest='verbose', help='inject test delta sequence')
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

# Load sync model
frame_sync_nn = FrameSyncNet(latent_dim)
frame_sync_nn.load_state_dict(torch.load(args.frame_sync_model_name,weights_only=True,map_location=torch.device('cpu')))
frame_sync_nn.eval()

M = model.M
Ncp = model.Ncp
Ns = model.Ns           # number of rate Rs symbols per modem frame
Nmf = int(Ns*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc
w = model.w.cpu().detach().numpy()
Fs = float(model.Fs)
alpha = 0.95

# load rx rate_Fs samples
rx = np.fromfile(args.rx, dtype=np.csingle)*args.gain
w_off = 2*np.pi*args.freq_offset/Fs
rx = rx*np.exp(-1j*w_off*np.arange(len(rx)))

# optional AGC, just a basic block based algorithm to get us started
if args.agc:
   # target RMS level is PAPR ~ 3 dB less than peak of 1.0
   target = 1.0*10**(-3/20)
   gain = target/np.sqrt(np.mean(np.abs(rx)**2))
   print(f"AGC target {target:3.2f} gain: {gain:3.2e}")
   rx *= gain
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

# BPF to remove some of the noise and improve acquisition
Ntap = 0
if args.bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   bpf = complex_bpf(Ntap, model.Fs, bandwidth, centre, len(rx))
   rx = bpf.bpf(rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[1].axis([0,len(rx)/model.Fs,0,3000])
   ax[1].set_title('After BPF')
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

# Acquisition - timing, freq offset, and signal present estimates

sequence_length = len(rx)//(Ncp+M) - 2
print(sequence_length)

# Normalised autocorrelation function
Ry_norm = np.zeros((sequence_length,Ncp+M),dtype=np.complex64)
for s in np.arange(sequence_length):
   for delta_hat in np.arange(Ncp+M):
      st = (s+1)*(Ncp+M) + delta_hat
      y_cp = rx[st-Ncp:st]
      y_m = rx[st-Ncp+M:st+M]
      Ry = np.dot(y_cp, np.conj(y_m))
      D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m))
      Ry_norm[s,delta_hat] = 2.*Ry/np.abs(D)

if len(args.write_Ry_norm):
   Ry_norm.flatten().tofile(args.write_Ry_norm)

# IIR smoothing
Ry_smooth = np.zeros((sequence_length,Ncp+M),dtype=np.complex64)
#Ry_smooth[0,:] = Ry_norm[0,:]
for s in np.arange(1,sequence_length):
   Ry_smooth[s,:] = Ry_smooth[s-1,:]*alpha + Ry_norm[s,:]*(1.-alpha)
if len(args.write_Ry_smooth):
   Ry_smooth.flatten().tofile(args.write_Ry_smooth)

# extract timing and freq offset estimates, signal detection flag
Ry_max = np.max(np.abs(Ry_smooth), axis=1)
if args.fix_delta_hat:
   delta_hat = args.fix_delta_hat*np.ones(sequence_length, dtype=np.int16)
else:
   delta_hat = np.int16(np.argmax(np.abs(Ry_smooth), axis=1))
Ts=0.42
sig_det = np.int16(Ry_max > Ts)

# post process to smooth out symbol-symbol variations in timing
# due to noise, but allowing big shifts during acquisition and 
# tracking of slow timing changes due to clock offsets
delta_hat_pp = np.zeros(sequence_length,dtype=np.float32)
count = 0
beta = 0.999
thresh = Ncp//2
for s in np.arange(1,sequence_length):
   if np.abs(delta_hat[s]-delta_hat_pp[s-1]) > thresh:
      count += 1
      if count > 5:
         delta_hat_pp[s] = delta_hat[s]
         count = 0
      else:
         delta_hat_pp[s] = delta_hat_pp[s-1]*beta + np.float32(delta_hat[s])*(1-beta)
   else:
      count = 0
      delta_hat_pp[s] = delta_hat_pp[s-1]*beta + np.float32(delta_hat[s])*(1-beta)

# (raw) freq offset estimates
freq_offset = np.zeros(sequence_length,dtype=np.float32)
for s in np.arange(1,sequence_length):
   delta_phi = np.angle(Ry_smooth[s,delta_hat[s]])
   freq_offset[s] = -delta_phi*Fs/(2.*np.pi*M)

if len(args.write_delta_hat):
   delta_hat.tofile(args.write_delta_hat)
if len(args.write_delta_hat_pp):
   np.int16(delta_hat_pp).tofile(args.write_delta_hat_pp)
if len(args.write_Ry_max):
   Ry_max.tofile(args.write_Ry_max)
if len(args.write_sig_det):
   sig_det.tofile(args.write_sig_det)
if len(args.write_freq_offset):
   freq_offset.tofile(args.write_freq_offset)
# optionally read in external timing est, overrides internal estimator   
if len(args.read_delta_hat):
   delta_hat = np.fromfile(args.read_delta_hat, dtype=np.float32)

# sync state machine, smooth freq offset ests when in sync
state = "noise"
count = 0
freq_offset_smooth = np.zeros(sequence_length,dtype=np.float32)
state_log = np.zeros(sequence_length,dtype=np.int16)
frame_sync_log = np.zeros((sequence_length,2),dtype=np.float32)
frame_sync_even = 0.
frame_sync_odd = 0.

# off air samples for i-th frame
rx_i = torch.zeros((Ns*(Ncp+M)),dtype=torch.complex64)

rx_phase = 1 + 1j*0
rx_phase_vec = np.zeros(Ncp+M,np.csingle)
Nframes = sequence_length//model.Ns
z_hat = torch.zeros((1,sequence_length, model.latent_dim), dtype=torch.float32)
i = 0

for s in np.arange(1,sequence_length):

   prev_state = state
   next_state = state

   if state == "noise":
      state_log[s] = 0
      if sig_det[s]:
         count += 1
         if count == 5:
            next_state = "signal"
            count = 0
            if args.nofreq_offset:
               delta_phi = 0.
            else:
               delta_phi = np.angle(Ry_smooth[s,delta_hat[s]])
            freq_offset_smooth[s] = -delta_phi*Fs/(2.*np.pi*M)
            frame_sync_even = 0.
            frame_sync_odd = 0.
   if state == "signal":
      state_log[s] = 1
      if not sig_det[s]:
         count += 1
         if count == args.hangover:
            next_state = "noise"
            count = 0
      else:
         count = 0
      if args.nofreq_offset:
         delta_phi = 0.
      else:
         delta_phi = np.angle(Ry_smooth[s,delta_hat[s]])
      freq_offset_smooth[s] = beta*freq_offset_smooth[s-1] - (1-beta)*delta_phi*Fs/(2.*np.pi*M)
      
      # correct freq offset
      #  keep phase vector normalised
      # extract single symbol, construct a frame with previous symbol

      # adjust timing to point to start of symbol
      delta_hat_rx = int(delta_hat_pp[s]-Ncp)

      # set up phase continous vector to correct freq offset
      freq_offset_rx = freq_offset_smooth[s]
      w = 2*np.pi*freq_offset_rx/Fs
      for n in range(Ncp+M):
         rx_phase = rx_phase*np.exp(-1j*w)
         rx_phase_vec[n] = rx_phase

      # extract symbol into end of i-th frame
      st = s*(Ncp+M) + delta_hat_rx
      en = st + Ncp+M
      rx_i[:Ncp+M] = rx_i[Ncp+M:]
      rx_i[Ncp+M:] = torch.tensor(rx_phase_vec*rx[st:en], dtype=torch.complex64)
      # run receiver to extract i-th freq domain OFDM symbols z_hat for one frame
      # Note this is run at symbol rate (twice frame rate) so we can get odd and even stats
      az_hat = model.receiver(rx_i,run_decoder=False)

      # update odd and even frame sync metrics
      frame_sync_metric_torch = frame_sync_nn(az_hat)
      frame_sync_metric = float(frame_sync_metric_torch[0,0,0])
      
      gamma = beta
      if s % 2:
         # odd frame alignment
         frame_sync_odd = gamma*frame_sync_odd + (1-gamma)*frame_sync_metric
         if frame_sync_odd > frame_sync_even:
            z_hat[0,i,:] = az_hat
            i += 1
      else:
         # even frame alignment
         frame_sync_even = gamma*frame_sync_even + (1-gamma)*frame_sync_metric
         if frame_sync_even > frame_sync_odd:
            z_hat[0,i,:] = az_hat
            i += 1

   state = next_state 
   
   frame_sync_log[s,0] = frame_sync_even
   frame_sync_log[s,1] = frame_sync_odd

   if args.verbose or state != prev_state:
      print(f"{s:3d} {i:3d} state: {state:6s} sig_det: {sig_det[s]:1d} count: {count:1d} ", end='', file=sys.stderr)
      print(f"fs: {frame_sync_odd > frame_sync_even:d} ", end='', file=sys.stderr)
      print(f"delta_hat: {delta_hat[s]:3.0f} delta_hat_pp: {delta_hat_pp[s]:3.0f} ", end='',file=sys.stderr)
      print(f"f_off: {freq_offset_smooth[s]:5.2f}", file=sys.stderr)

# truncate from max length
z_hat = z_hat[:,:i,:]

if len(args.write_freq_offset_smooth):
   freq_offset_smooth.tofile(args.write_freq_offset_smooth)
if len(args.write_state):
   state_log.tofile(args.write_state)
if len(args.write_frame_sync):
   frame_sync_log.flatten().tofile(args.write_frame_sync)

rx = np.concatenate((rx,np.zeros(Ncp+M,dtype=np.complex64)))

"""
# use timing estimates as they evolve to extract frames
Nframes = sequence_length//model.Ns
z_hat = torch.zeros((1,Nframes, model.latent_dim), dtype=torch.float32)
delta_hat_rx = np.zeros(Nframes,dtype=np.int16)
rx_phase = 1 + 1j*0
rx_phase_vec = np.zeros(model.Ns*(Ncp+M),np.csingle)

# note only one time estimate per frame (Ns symbols), we don't want a timing change
# mid frame
for i in np.arange(0,Nframes):
   # map delta_hat from Ncp/M junction to start sample of symbol
   delta_hat_rx[i] = int(delta_hat_pp[model.Ns*i]-Ncp)
   
   # set up phase continous vector to correct freq offset
   freq_offset_rx = freq_offset_smooth[model.Ns*i]
   w = 2*np.pi*freq_offset_rx/Fs
   for n in range(model.Ns*(Ncp+M)):
      rx_phase = rx_phase*np.exp(-1j*w)
      rx_phase_vec[n] = rx_phase

   st = (model.Ns*i)*(Ncp+M) + delta_hat_rx[i]
   st = max(st,0)
   en = st + model.Ns*(Ncp+M)
   if i < 10:
      print(i,delta_hat_rx[i],st,en)
   # extract rx samples for i-th frame
   rx_i = torch.tensor(rx_phase_vec*rx[st:en], dtype=torch.complex64)
   # run receiver to extract i-th freq domain OFDM symbols z_hat
   az_hat = model.receiver(rx_i,run_decoder=False)
   z_hat[0,i,:] = az_hat
"""
print("z_hat.shape",z_hat.shape)

if len(args.write_delta_hat_rx):
   np.float32(delta_hat_rx).flatten().tofile(args.write_delta_hat_rx)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)

"""
# now perform ML frame sync, two possibilities offset by one OFDM symbol (half a z vector)
if args.noframe_sync == False:
   Nsync_syms = 10 # average sync metric over this many OFDM symbols
   print(z_hat.shape)
   z_hat = torch.reshape(z_hat,(1,-1,latent_dim//2))
   print(z_hat.shape)
   sync_st=10
   sync_even = torch.mean(frame_sync_nn(torch.reshape(z_hat[0,sync_st:sync_st+Nsync_syms,:],(1,-1,latent_dim))))
   sync_odd = torch.mean(frame_sync_nn(torch.reshape(z_hat[0,sync_st+1:sync_st+1+Nsync_syms,:],(1,-1,latent_dim))))
   print(f"sync_even: {sync_even:5.2f} sync_odd: {sync_odd:5.2f}")
   if sync_even > sync_odd:
      offset = 0
   else:
      offset = 1
   z_hat_len = z_hat.shape[1]
   z_hat = z_hat[:,offset:z_hat_len-offset,:]
   z_hat = torch.reshape(z_hat,(1,-1,latent_dim))
"""

# run RADE decoder
features_hat = model.core_decoder(z_hat)
features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:nb_total_features-num_features]], dim=-1)
#print(features_hat.shape)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)
