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
import torch
from radae import RADAE

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--pilots', action='store_true', help='insert pilot symbols')
parser.add_argument('--ber_test', action='store_true', help='send random PSK bits through channel model, measure BER')
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, pilots=args.pilots)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

# load rx rate Fs samples
rx = torch.tensor(np.fromfile(args.rx, dtype=np.csingle))

# acquisition - coarse & fine timing
"""
M = model.get_M()
Ns = model.get_Ns()
Nmf = Ns*M                # number of samples in one modem frame
p = model.p               # pilot sequence
D = np.zeros(2*Nmf)       # correlation at various time offsets
Dtmax = 0
tmax = 0
Pthresh = 0.9
acquired = False
while not acquired and len(rx) >= Nmf+M:
   # search modem frame for maxima
   for t in range(Nmf):
      D[t] = rx[t:t+model.M].H*p
      if np.abs(D[t]) > Dtmax:
         Dtmax = np.abs(D[t])
         tmax = t
   
   sigma_est = np.std(D)
   Dthresh = sigma_est*np.sqrt(-np.log(Pthresh))
   print(f"Dthresh: {Dthresh:f} Dtmax: {Dtmax:f} tmax: {tmax:d}")
   if Dtmax > Dthresh:
      acquired = True
      print("Acquired!")
   else:
      # advance one frame and search again
      rx = rx[Nmf:-1]
if not acquired:
   print("Acquisition failed....")
   quit()

rx = rx[tmax:-1]

# magnitude normalisation
r = rx[:M]
g = r.H*r/(p.H*p)
print(f"g: {g:f}")
rx = rx/g
"""
# push model to device and run receiver
model.to(device)
rx = rx.to(device)
features_hat, z_hat = model.receiver(rx)

features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

# write real valued latent vectors
if len(args.write_latent):
   z_hat = z_hat.cpu().detach().numpy().flatten().astype('float32')
   z_hat.tofile(args.write_latent)

