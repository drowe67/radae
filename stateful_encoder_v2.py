"""
   RADE V2 stateful encoder/decoder reference script.
   Used for testing the C port of CoreEncoderStatefull and CoreDecoderStatefull.

   Usage (encoder test):
     python3 stateful_encoder_v2.py model.pth features_in.f32 features_hat.f32 \
         --read_latent z_c.f32 --loss_test 0.3

   Usage (decoder test, write Python latents for C decoder to consume):
     python3 stateful_encoder_v2.py model.pth features_in.f32 features_hat.f32 \
         --write_latent z_py.f32

/*
   Copyright (C) 2025 David Rowe

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
from radae import RADAE, distortion_loss

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='path to V2 model checkpoint (.pth)')
parser.add_argument('features', type=str, help='path to input feature file (.f32, 36 floats/frame from lpcnet_demo)')
parser.add_argument('features_hat', type=str, help='path to output decoded feature file (.f32), or /dev/null')
parser.add_argument('--latent-dim', type=int, default=56, help='latent dimension (default: 56)')
parser.add_argument('--w1-dec', type=int, default=128, help='decoder hidden width (default: 128)')
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to threshold, print PASS/FAIL')
parser.add_argument('--read_latent', type=str, help='read latents from file (e.g. C encoder output) instead of running Python encoder')
parser.add_argument('--write_latent', type=str, help='write Python encoder latents to file (e.g. for C decoder testing)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ""

nb_total_features = 36
num_used_features = 20   # lpcnet_demo produces 36, first 20 are LPCNet features
num_features      = 21   # V2 uses 20 + 1 auxdata symbol

latent_dim = args.latent_dim
w1_dec     = args.w1_dec

# Load V2 model (mirrors the weight-loading pattern from rx2.py / export_rade_v2_weights.py)
model = RADAE(num_features, latent_dim, EbNodB=100, Nzmf=1,
              rate_Fs=True, bottleneck=0, cyclic_prefix=0.004,
              w1_dec=w1_dec, w1_dec_stateful=w1_dec, peak=True)

checkpoint  = torch.load(args.model_name, map_location='cpu', weights_only=True)
state_dict  = checkpoint['state_dict']
model_dict  = model.state_dict()
pretrained  = {k: v for k, v in state_dict.items()
               if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained)
model.load_state_dict(model_dict, strict=False)
model.core_encoder_statefull_load_state_dict()
model.core_decoder_statefull_load_state_dict()
model.eval()

# Load features (36 floats/frame from lpcnet_demo), use first 20, append auxdata=-1
features_raw = np.reshape(np.fromfile(args.features, dtype=np.float32), (-1, nb_total_features))
nb_rounded   = model.num_10ms_times_steps_rounded_to_modem_frames(features_raw.shape[0])
features_raw = features_raw[:nb_rounded, :num_used_features]
# Append auxdata column (-1 signals normal TX)
auxdata      = -np.ones((nb_rounded, 1), dtype=np.float32)
features_in  = np.concatenate([features_raw, auxdata], axis=1)   # (T, 21)
features     = torch.tensor(features_in).unsqueeze(0)            # (1, T, 21)
print(f"Processing: {nb_rounded} feature vectors")

enc = model.core_encoder_statefull.module
dec = model.core_decoder_statefull.module

if args.read_latent:
    # Use C encoder output instead of Python encoder
    z_np = np.fromfile(args.read_latent, dtype=np.float32).reshape(1, -1, latent_dim)
    z    = torch.tensor(z_np[:, :nb_rounded // model.enc_stride, :])
else:
    # Run Python stateful encoder one enc_stride at a time
    z_list = []
    with torch.no_grad():
        for i in range(0, nb_rounded, model.enc_stride):
            feat_i = features[:, i:i+model.enc_stride, :]
            z_list.append(enc(feat_i))
    z = torch.cat(z_list, dim=1)   # (1, T/enc_stride, latent_dim)

if args.write_latent:
    z.detach().numpy().astype(np.float32).flatten().tofile(args.write_latent)
    print(f"Wrote {z.shape[1]} latent vectors to {args.write_latent}")

# Run Python stateful decoder
features_hat_list = []
with torch.no_grad():
    for i in range(z.shape[1]):
        z_i = z[:, i:i+1, :]
        features_hat_list.append(dec(z_i))
features_hat = torch.cat(features_hat_list, dim=1)   # (1, T, 21)

loss = distortion_loss(features[:, :, :num_used_features],
                       features_hat[:, :, :num_used_features]).item()
print(f"loss: {loss:.3f}")
if args.loss_test > 0.0:
    if loss < args.loss_test:
        print("PASS")
    else:
        print("FAIL")

# Write decoded features padded to 36 floats/frame for downstream tools
features_hat_padded = torch.cat(
    [features_hat[:, :, :num_used_features],
     torch.zeros(1, features_hat.shape[1], nb_total_features - num_used_features)], dim=-1)
features_hat_padded.detach().numpy().astype(np.float32).flatten().tofile(args.features_hat)
