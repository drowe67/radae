"""

  Radio Autoencoder streaming transmitter: 
  
  features in, rate Fs IQ.f32 out.

  Designed to connected to a SDR to perform real time RADAE decoding on 
  received sample streams.  Full function state machine and continous 
  updates of timing, freq offsets and amplituide estimates.
  
  Copyright (c) 2024 by David Rowe

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

import os, sys, argparse, struct
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,transmitter_one

parser = argparse.ArgumentParser(description='RADAE streaming transmitter, features.f32 on stdin, IQ.f32 stdout')

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--ber_test', type=str, default="", help='symbols are PSK bits, compare to z.f32 file to calculate BER')
parser.add_argument('--bottleneck', type=int, default=3, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--no_stdout', action='store_false', dest='use_stdout', help='disable the use of stdout (e.g. with python3 -m cProfile)')
parser.set_defaults(use_stdout=True)
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, 
              pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
              coarse_mag=True,time_offset=-16, bottleneck=args.bottleneck)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.core_encoder_statefull_load_state_dict()
model.eval()

transmitter = transmitter_one(model.latent_dim,model.enc_stride,model.Nzmf,model.Fs,model.M,model.Ncp,
                              model.Winv,model.Nc,model.Ns,model.w,model.P,model.bottleneck,model.pilot_gain)

nb_floats = model.Nzmf*model.enc_stride*nb_total_features

with torch.inference_mode():
   while True:
      buffer = sys.stdin.buffer.read(nb_floats*struct.calcsize("f"))
      #print(len(buffer), file=sys.stderr)
      if len(buffer) != nb_floats*struct.calcsize("f"):
         break
      buffer_f32 = np.frombuffer(buffer,np.single)
      features = torch.reshape(torch.tensor(buffer_f32),(1,model.Nzmf*model.enc_stride, nb_total_features))
      features = features[:,:,:num_used_features]
      #print(features.shape, file=sys.stderr)
      num_timesteps_at_rate_Rs = model.num_timesteps_at_rate_Rs(model.Nzmf*model.enc_stride)
      z = model.core_encoder_statefull(features)
      tx = transmitter.transmitter_one(z,num_timesteps_at_rate_Rs)
      tx = tx.cpu().detach().numpy().flatten().astype('csingle')
      if args.use_stdout:
         sys.stdout.buffer.write(tx)

if args.use_stdout:
   eoo = model.eoo
   eoo = eoo.cpu().detach().numpy().flatten().astype('csingle')
   sys.stdout.buffer.write(eoo)
