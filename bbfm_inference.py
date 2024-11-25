"""
/* Copyright (c) 2024 modifications for radio autoencoder project
   by David Rowe */
   
/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
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

from radae import BBFM, distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--cuda-visible-devices', type=str, help="set to 0 to run using GPU rather than CPU", default="")
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--CNRdB', type=float, default=100, help='FM demod input CNR in dB')
parser.add_argument('--passthru', action='store_true', help='copy features in to feature out, bypassing ML network')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs fading channel magnitude samples, rate Rs time steps by Nc=1 carriers .f32 format')
parser.add_argument('--write_CNRdB', type=str, default="", help='path to output file of CNRdB per sample after fading in .f32 format')
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to arg, print PASS/FAIL')
args = parser.parse_args()

# set visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

latent_dim = args.latent_dim

# not exposed
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = BBFM(num_features, latent_dim, args.CNRdB)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
checkpoint['state_dict'] = model.state_dict()

# load features from file
feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
features = features_in[:,:nb_features_rounded,:]
features = features[:, :, :num_used_features]
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded} feature vectors")

# default rate Rb multipath model H=1
Rb = model.Rb
Nc = 1
num_timesteps_at_rate_Rs = model.num_timesteps_at_rate_Rs(nb_features_rounded)
H = torch.ones((1,num_timesteps_at_rate_Rs,Nc))

# user supplied rate Rs multipath model, sequence of H magnitude samples
if args.h_file:
   H = np.reshape(np.fromfile(args.h_file, dtype=np.float32), (1, -1, Nc))
   print(H.shape, num_timesteps_at_rate_Rs)
   if H.shape[1] < num_timesteps_at_rate_Rs:
      print("Multipath H file too short")
      quit()
   H = H[:,:num_timesteps_at_rate_Rs,:]
   H = torch.tensor(H)

if __name__ == '__main__':

   if args.passthru:
      features_hat = features_in.flatten()
      features_hat.tofile(args.features_hat)
      quit()

   # push model to device and run test
   model.to(device)
   features = features.to(device)
   H = H.to(device)
   output = model(features,H)

   # Lets check actual SNR at output of FM demod
   tx_sym = output["z_hat"].cpu().detach().numpy()
   S = np.mean(np.abs(tx_sym)**2)
   N = np.mean(output["sigma"].cpu().detach().numpy()**2)
   SNRdB_meas = 10*np.log10(S/N)
   print(f"SNRdB Measured: {SNRdB_meas:6.2f}")

   features_hat = output["features_hat"][:,:,:num_used_features]
   features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
   features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
   features_hat.tofile(args.features_hat)

   loss = distortion_loss(features,output['features_hat']).cpu().detach().numpy()[0]
   print(f"loss: {loss:5.3f}")
   if args.loss_test > 0.0:
      if loss < args.loss_test:
         print("PASS")
      else:
         print("FAIL")

   # write output symbols (latent vectors)
   if len(args.write_latent):
      z_hat = output["z_hat"].cpu().detach().numpy().flatten().astype('float32')
      z_hat.tofile(args.write_latent)
   
   # write CNRdB after fading
   if len(args.write_CNRdB):
      CNRdB = output["CNRdB"].cpu().detach().numpy().flatten().astype('float32')
      CNRdB.tofile(args.write_CNRdB)
      
