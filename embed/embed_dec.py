"""

   Basic decoder to test operation of RADAE using Python Embedding.

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

import os, sys
import numpy as np
import torch
sys.path.append("../")
from radae import RADAE, distortion_loss

# Hard code all this for now to avoid arg passing complexities
model_name = "../model05/checkpoints/checkpoint_epoch_100.pth"
features_in_fn = "features_in.f32"
features_out_fn = "features_out.f32"
latent_dim = 80
auxdata = False

os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")
nb_total_features = 36
num_features = 20
num_used_features = 20
if auxdata:
    num_features += 1

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, 100.0,)
checkpoint = torch.load(model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

def my_decode():
   # dataloader
   features_in = np.reshape(np.fromfile(features_in_fn, dtype=np.float32), (1, -1, nb_total_features))
   nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
   features = torch.tensor(features_in[:,:nb_features_rounded,:num_used_features])
   print(f"Processing: {nb_features_rounded} feature vectors")

   model.to(device)
   features = features.to(device)
   z = model.core_encoder(features)
   features_hat = model.core_decoder(z)
   
   loss = distortion_loss(features,features_hat).cpu().detach().numpy()[0]
   print(f"loss: {loss:5.3f}")

   features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
   features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
   features_hat.tofile(features_out_fn)

if __name__ == '__main__':
   my_decode()