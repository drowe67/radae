"""

   Compares stateful encoder (that operates one frame at a time) to 
   vanilla encoder.  Also used for testing C port of encoder.

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

from radae import RADAE, distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to arg, print PASS/FAIL')
parser.add_argument('--read_latent', type=str, help='path to optional input latent file z.f32, e.g. for C encoder testing')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")
latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, 100.0)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
# Stateful encoder wasn't present during training, so we need to load weights from existing encoder
model.core_encoder_statefull_load_state_dict()

model2 = RADAE(num_features, latent_dim, 100.0)
model2.load_state_dict(checkpoint['state_dict'], strict=False)

# dataloader
features_in = np.reshape(np.fromfile(args.features, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
features = torch.tensor(features_in[:,:nb_features_rounded,:num_used_features])
print(f"Processing: {nb_features_rounded} feature vectors")

if __name__ == '__main__':

   model.to(device)
   features = features.to(device)

   # vanilla encoder that doesn't preserve state
   z = model.core_encoder(features)
   print(z.shape)

   if args.read_latent:
      z_in = np.reshape(np.fromfile(args.read_latent, dtype=np.float32), (1, -1, args.latent_dim))
      z_statefull = torch.tensor(z_in[:,:nb_features_rounded//4,:])
   else:
      # Python encoder that preserves state, test by processing 12 feature vecs on each call 
      z_statefull = torch.empty(1,0,model.latent_dim)
      step = 3
      for i in range(0,features.shape[1],model.enc_stride*step):
         z_one = model.core_encoder_statefull(features[:,i:i+model.enc_stride*step,:])
         z_statefull = torch.cat([z_statefull, z_one],dim=1) 
   
   # test with vanilla decoder
   features_hat = model.core_decoder(z)
   features_hat_statefull = model2.core_decoder(z_statefull)
      
   loss = distortion_loss(features,features_hat).cpu().detach().numpy()[0]
   loss_statefull = distortion_loss(features,features_hat_statefull).cpu().detach().numpy()[0]
   loss_delta =  distortion_loss(features_hat,features_hat_statefull).cpu().detach().numpy()[0]
   print(f"loss: {loss:5.3f} {loss_statefull:5.3f} {loss_delta:5.3f}")
   if args.loss_test > 0.0:
      if loss_statefull < args.loss_test and loss_delta < 0.01:
         print("PASS")
      else:
         print("FAIL")

   features_hat_statefull = torch.cat([features_hat_statefull, torch.zeros_like(features_hat_statefull)[:,:,:16]], dim=-1)
   features_hat_statefull = features_hat_statefull.cpu().detach().numpy().flatten().astype('float32')
   features_hat_statefull.tofile(args.features_hat)
