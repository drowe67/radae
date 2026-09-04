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
from radae_v2 import RADEv2Receiver


parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('frame_sync_model_name', type=str, help='path to frame sync model in .pth format')
parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 56", default=56)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--cp', type=float, default=0.004, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0.04)')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--freq_offset', type=float, default=0, help='correct for this frequency offset')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling time offset in samples')
parser.add_argument('--correct_time_offset', type=int, default=-8, help='introduces a delay (or advance if -ve) in samples, applied in freq domain (default -8)')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--acq_time_target', type=float, default=1.0, help='Acquisition test mode mean acquisition time target (default 1.0)')
parser.add_argument('--stateful',  action='store_true', help='use stateful core decoder')
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation (fine timing)',default = 160,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension (fine timing)',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension (fine timing)',default = 160,required = False)
parser.add_argument('--write_Ry_smooth', type=str, default="", help='path to smoothed autocorrelation output feature file dim (seq_len,Ncp+M) .c64 format')
parser.add_argument('--write_delta_hat', type=str, default="", help='path to delta_hat output file dim (seq_len) in .float32 format')
parser.add_argument('--write_delta_hat_g', type=str, default="", help='path to delta_hat_g output file dim (seq_len) in .float32 format')
parser.add_argument('--write_Ry_max', type=str, default="", help='path to Ty_max output file dim (seq_len) in .f32 format')
parser.add_argument('--write_sig_det', type=str, default="", help='path to signal detection flag output file dim (seq_len) in .int16 format')
parser.add_argument('--write_freq_offset', type=str, default="", help='path to freq offset est output file dim (seq_len) in .float32 format')
parser.add_argument('--write_delta_hat_rx', type=str, default="", help='path to delta_hat_rx file dim (seq_len) in .f32 format')
parser.add_argument('--write_state', type=str, default="", help='path to sync state machine output file dim (seq_len) in .int16 format')
parser.add_argument('--write_frame_sync', type=str, default="", help='path to frame sync output file dim (seq_len,2) in .int16 format')
parser.add_argument('--read_delta_hat', type=str, default="", help='path to delta_hat input file dim (seq_len) in .f32 format')
parser.add_argument('--fix_delta_hat', type=int,  default=0, help='disable timing estimation and used fixed delta_hat (default: use timing estimation)')
parser.add_argument('--write_gain', type=str, default="", help='path to AGC output file dim (seq_len) .f32 format')
parser.add_argument('--write_snr_est', type=str, default="", help='path to SNR estimate output file dim (seq_len) .f32 format (dB)')
parser.set_defaults(bpf=True)
parser.set_defaults(auxdata=True)
parser.set_defaults(verbose=True)
parser.add_argument('--pad_samples', type=int, default=0, help='Pad input with samples to simulate different timing offsets in rx signal')
parser.add_argument('--gain', type=float, default=1.0, help='manual gain control')
parser.add_argument('--agc', action='store_true', help='automatic gain control')
parser.add_argument('--w1_dec', type=int, default=128, help='Decoder GRU output dimension (default 128)')
parser.add_argument('--nofreq_offset', action='store_true', help='disable freq offset correction (default enabled)')
parser.add_argument('--test_mode', action='store_true', help='inject test delta sequence')
parser.add_argument('--hangover', type=int, default=75, help='Number of symbols of no signal before returning to noise state (default 75)')
parser.add_argument('--quiet', action='store_false', dest='verbose', help='inject test delta sequence')
parser.add_argument('--verbose', action='store_true', dest='verbose', help='inject test delta sequence')
parser.add_argument('--stop_at', type=int, default=0, help='exit program after this many symbols (default disabled)')
parser.add_argument('--timing_adj_at', type=int, default=0, help='enable timing adjust after this many symbols (default disabled)')
parser.add_argument('--reset_output_on_resync', action='store_true', help='only keep output from last resync (default disabled)')
parser.add_argument('--write_features', type=str, default="", help='path to write decoder output features (nb_total_features) in .f32 format')
parser.set_defaults(limit_pitch=True)
parser.add_argument('--nolimit_pitch', action='store_false', dest='limit_pitch', help='disable limiting (clip) lower end of pitch feature to prevent synthesis pops with some speakers/channels (default enabled)')
parser.set_defaults(mute=False)
parser.add_argument('--mute', action='store_false',  dest='mute', help='enable mute when sig lost (default disabled)')
parser.set_defaults(eoo=True)
parser.add_argument('--no_eoo', action='store_false', dest='eoo', help='disable EOO (end of over) detection (default enabled)')
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
              rate_Fs=True, bottleneck=0, cyclic_prefix=args.cp,
              time_offset=args.time_offset, correct_time_offset=args.correct_time_offset,
              stateful_decoder=args.stateful, w1_dec=args.w1_dec, w1_dec_stateful=args.w1_dec)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)

# model was trained with core_decoder_stateful with different w1_dec_stateful.  So we remove
# mismatched size entries from dictionary so we can load model
state_dict = checkpoint['state_dict']
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
model.core_decoder_statefull_load_state_dict()

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

# load rx rate_Fs samples
rx = np.fromfile(args.rx, dtype=np.csingle)*args.gain
w_off = 2*np.pi*args.freq_offset/Fs
rx = rx*np.exp(-1j*w_off*np.arange(len(rx)))

# ensure an integer number of frames
rx = np.concatenate((np.zeros(args.pad_samples, dtype=np.complex64),rx))

rx = rx[:Nmf*(len(rx)//Nmf)]
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)//Nmf}", file=sys.stderr)

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
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
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

sequence_length = len(rx)//(Ncp+M)
print(sequence_length, file=sys.stderr)

receiver = RADEv2Receiver(model, frame_sync_nn, args)
z_hat        = torch.zeros((1, sequence_length, model.latent_dim), dtype=torch.float32)
features_hat = torch.zeros((1, sequence_length * model.dec_stride, num_features))

# Diagnostic logs (indexed by symbol number)
sl = sequence_length
sym_len = receiver.sym_len
state_log       = np.zeros(sl, dtype=np.int16)
frame_sync_log  = np.zeros((sl, 2), dtype=np.float32)
Ry_norm_log     = np.zeros((sl, sym_len), dtype=np.complex64)
Ry_smooth_log   = np.zeros((sl, sym_len), dtype=np.complex64)
sig_det_log     = np.zeros(sl, dtype=np.int16)
delta_hat_log   = np.zeros(sl, dtype=np.float32)
delta_hat_g_log = np.zeros(sl, dtype=np.float32)
freq_offset_log = np.zeros(sl, dtype=np.float32)
gain_log        = np.zeros(sl, dtype=np.float32)
snr_est_dB_log  = np.zeros(sl, dtype=np.float32)

nin = receiver.sym_len
prx = 0

while prx + nin < len(rx):
   receiver.s += 1
   st, en = prx, prx + nin
   prx   += nin

   prev_state = receiver.state
   next_state, features_hat_slice, nin, sig_det, sine_det, gain = receiver._process_symbol(rx[st:en], nin)

   s = receiver.s
   if s < sl:
      state_log[s]         = 0 if receiver.state == "idle" else 1
      gain_log[s]          = gain
      snr_est_dB_log[s]    = receiver.snr_est_dB
      Ry_norm_log[s]       = receiver.Ry_norm
      Ry_smooth_log[s]     = receiver.Ry_smooth
      sig_det_log[s]       = sig_det
      delta_hat_log[s]     = receiver.delta_hat
      delta_hat_g_log[s]   = receiver.delta_hat_g
      freq_offset_log[s]   = receiver.freq_offset
      frame_sync_log[s, 0] = receiver.frame_sync_even
      frame_sync_log[s, 1] = receiver.frame_sync_odd

   if receiver.args.verbose or receiver.state != prev_state:
      receiver._print_status(sig_det, sine_det, nin)

   receiver.state = next_state

   if features_hat_slice is not None:
      z_hat[0, receiver.i, :] = receiver.az_hat
      dec_st = receiver.model.dec_stride * receiver.i
      dec_en = receiver.model.dec_stride * (receiver.i + 1)
      features_hat[0, dec_st:dec_en, :] = features_hat_slice
      receiver.i += 1

   if receiver.s > receiver.args.timing_adj_at:
      receiver.timing_adj = 1

   if receiver.s == receiver.args.stop_at:
      quit()

z_hat        = z_hat[:, :receiver.i, :]
features_hat = features_hat[:, :receiver.i * receiver.model.dec_stride, :]

if len(args.write_Ry_smooth):
   Ry_smooth_log.flatten().tofile(args.write_Ry_smooth)
if len(args.write_delta_hat):
   delta_hat_log.tofile(args.write_delta_hat)
if len(args.write_delta_hat_g):
   delta_hat_g_log.tofile(args.write_delta_hat_g)
if len(args.write_sig_det):
   sig_det_log.tofile(args.write_sig_det)
if len(args.write_freq_offset):
   freq_offset_log.tofile(args.write_freq_offset)
if len(args.write_gain):
   gain_log.tofile(args.write_gain)
if len(args.write_snr_est):
   snr_est_dB_log.tofile(args.write_snr_est)
if len(args.write_state):
   state_log.tofile(args.write_state)
if len(args.write_frame_sync):
   frame_sync_log.flatten().tofile(args.write_frame_sync)

rx = np.concatenate((rx,np.zeros(Ncp+M,dtype=np.complex64)))
z_hat.shape
print(f"n_acq: {receiver.n_acq:d}",file=sys.stderr)
print(f"latent vectors: {z_hat.shape[1]:d}",file=sys.stderr)

features_hat_out = np.zeros(0)
if z_hat.shape[1]:
   features_hat_out = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:nb_total_features-num_features]], dim=-1)
   features_hat_out = features_hat_out.cpu().detach().numpy().flatten().astype('float32')
features_hat_out.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)

if len(args.write_features):
   features_hat_out.tofile(args.write_features)
