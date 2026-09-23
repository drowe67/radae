"""
/*
  RADE V2 transmitter: speech features in, rate Fs IQ samples out.

  Stateful, streaming encoder: enc_stride feature vectors in, Ns OFDM symbols out per call.

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

import os
import argparse

import numpy as np
import torch

from radae import RADAE
from radae_v2 import RADEv2Transmitter

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('rx_out', type=str, help='path to output file of rate Fs IQ samples in ..IQIQ...f32 format')
parser.add_argument('--latent-dim', type=int, default=56, help='number of symbols produced by encoder (default 56)')
parser.add_argument('--cp', type=float, default=0.004, help='length of cyclic prefix in seconds (default 0.004)')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling offset in samples (default -16)')
parser.add_argument('--correct_time_offset', type=int, default=-8, help='freq domain time offset correction in samples (default -8)')
parser.add_argument('--w1_enc', type=int, default=64, help='encoder GRU output dimension (default 64)')
parser.add_argument('--w2_enc', type=int, default=96, help='encoder conv output dimension (default 96)')
parser.add_argument('--w1_dec', type=int, default=128, help='decoder GRU output dimension, needed for checkpoint loading (default 128)')
parser.add_argument('--w2_dec', type=int, default=32, help='decoder conv output dimension, needed for checkpoint loading (default 32)')
parser.add_argument('--write_latent', type=str, default="", help='path to output latent vectors z[latent_dim] in .f32 format')
parser.set_defaults(ssb_bpf=True)
parser.add_argument('--no_bpf', action='store_false', dest='ssb_bpf', help='disable SSB BPF (default enabled)')
parser.set_defaults(auxdata=True)
parser.add_argument('--no_auxdata', action='store_false', dest='auxdata', help='disable auxiliary data symbol (default enabled)')
parser.set_defaults(end_of_over_v2=True)
parser.add_argument('--no_eoo', action='store_false', dest='end_of_over_v2', help='disable end-of-over sequence (default enabled)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ""

latent_dim = args.latent_dim
nb_total_features = 36
num_used_features = 20
num_features = num_used_features
if args.auxdata:
    num_features += 1

model = RADAE(num_features, latent_dim, EbNodB=100, Nzmf=1, rate_Fs=True, bottleneck=0,
              cyclic_prefix=args.cp, time_offset=args.time_offset,
              correct_time_offset=args.correct_time_offset,
              w1_dec=args.w1_dec, w2_dec=args.w2_dec,
              w1_enc=args.w1_enc, w2_enc=args.w2_enc,
              ssb_bpf=args.ssb_bpf)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.core_encoder_statefull_load_state_dict()
model.eval()

feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
features = features_in[:, :nb_features_rounded, :num_used_features]
if args.auxdata:
    aux_symb = -np.ones((1, features.shape[1], 1), dtype=np.float32)
    features = np.concatenate([features, aux_symb], axis=2)
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded} feature vectors")

transmitter = RADEv2Transmitter(model)
enc_stride = model.enc_stride
num_frames = nb_features_rounded // enc_stride

tx = np.empty(0, dtype=np.csingle)
z_frames = []
for i in range(num_frames):
    feat_frame = features[:, i * enc_stride : (i + 1) * enc_stride, :]
    tx = np.concatenate([tx, transmitter.transmit_frame(feat_frame)])
    if len(args.write_latent):
        z_frames.append(transmitter.last_z.numpy().flatten())

if args.end_of_over_v2:
    tx = np.concatenate([tx, transmitter.eoo()])

tx.tofile(args.rx_out)

if len(args.write_latent):
    np.concatenate(z_frames).astype('float32').tofile(args.write_latent)
