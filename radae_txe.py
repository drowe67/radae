"""

  Radio Autoencoder streaming transmitter, "embedded" version.
  
  features in, rate Fs IQ.f32 out.
  
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

import os, sys, struct,argparse
import numpy as np
import torch
from radae import RADAE,transmitter_one,complex_bpf

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

nb_total_features = 36
num_used_features = 20

class radae_tx:
   def __init__(self, model_name, latent_dim=80, auxdata=True, bottleneck=3, txbpf_en=False):

      self.latent_dim = latent_dim
      self.auxdata = auxdata
      self.bottleneck = bottleneck
      self.txbpf_en = txbpf_en

      self.num_features = 20
      if auxdata:
         self. num_features += 1

      # load model from a checkpoint file
      self.model = RADAE(self.num_features, latent_dim, EbNodB=100, rate_Fs=True, 
                  pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
                  coarse_mag=True,time_offset=-16, bottleneck=bottleneck)
      model = self.model
      checkpoint = torch.load(model_name, map_location='cpu')
      model.load_state_dict(checkpoint['state_dict'], strict=False)
      model.core_encoder_statefull_load_state_dict()
      model.eval()

      self.transmitter = transmitter_one(model.latent_dim,model.enc_stride,model.Nzmf,model.Fs,model.M,model.Ncp,
                                         model.Winv,model.Nc,model.Ns,model.w,model.P,model.bottleneck,model.pilot_gain)
      if self.txbpf_en:
         Ntap=101
         w = np.array(model.w)
         Nc = model.Nc
         bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
         centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
         print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
         self.txbpf = complex_bpf(Ntap, model.Fs, bandwidth,centre)

      # number of input floats per processing frame (TOOD refactor to more sensible variable names)
      self.nb_floats = model.Nzmf*model.enc_stride*nb_total_features
      # number of output csingles per processing frame
      self.Nmf = int((model.Ns+1)*(model.M+model.Ncp))
      # number of output csingles for EOO frame
      self.Neoo = int((model.Ns+2)*(model.M+model.Ncp))

   def get_nb_floats(self):
      return self.nb_floats
   def get_Nmf(self):
      return self.Nmf
   def get_Neoo(self):
      return  self.Neoo

   def do_radae_tx(self,buffer_f32,tx_out):
      model = self.model

      with torch.inference_mode():
         features = torch.reshape(torch.tensor(buffer_f32),(1,model.Nzmf*model.enc_stride, nb_total_features))
         features = features[:,:,:num_used_features]
         if self.auxdata:
            aux_symb =  -torch.ones((1,features.shape[1],1))
            symb_repeat = 4
            for i in range(1,symb_repeat):
               aux_symb[0,i::symb_repeat,:] = aux_symb[0,::symb_repeat,:]
            features = torch.concatenate([features, aux_symb],axis=2)
         #print(features.shape, file=sys.stderr)
         num_timesteps_at_rate_Rs = model.num_timesteps_at_rate_Rs(model.Nzmf*model.enc_stride)
         z = model.core_encoder_statefull(features)
         tx = self.transmitter.transmitter_one(z,num_timesteps_at_rate_Rs)
         tx = tx.cpu().detach().numpy().flatten().astype('csingle')
         if self.txbpf_en:
            tx = self.txbpf.bpf(tx)
            tx = np.clip(abs(tx),a_min=0, a_max=1)*np.exp(1j*np.angle(tx))
         
         # not very Pythonic but works (TODO work out how to return numpy vecs to C)
         np.copyto(tx_out,tx)

   # send end of over frame
   def do_eoo(self,tx_out):
      eoo = self.model.eoo
      eoo = eoo.cpu().detach().numpy().flatten().astype('csingle')
      if self.txbpf_en:
         eoo = self.txbpf.bpf(eoo)
         eoo = np.clip(abs(eoo),a_min=0,a_max=1)*np.exp(1j*np.angle(eoo))
      np.copyto(tx_out,eoo)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='RADAE streaming trannsmnitter, features.f32 on stdit, IQ.f32 on output')
   parser.add_argument('--model_name', type=str, help='path to model in .pth format', default="model19_check3/checkpoints/checkpoint_epoch_100.pth")
   parser.add_argument('--noauxdata', dest="auxdata", action='store_false', help='disable injection of auxillary data symbols')
   parser.add_argument('--txbpf', action='store_true', help='enable Tx BPF')
   parser.set_defaults(auxdata=True)
   args = parser.parse_args()
   tx = radae_tx(model_name=args.model_name, auxdata=args.auxdata, txbpf_en=args.txbpf)

   tx_out = np.zeros(tx.Nmf,dtype=np.csingle)
   while True:
      buffer = sys.stdin.buffer.read(tx.nb_floats*struct.calcsize("f"))
      if len(buffer) != tx.nb_floats*struct.calcsize("f"):
         break
      buffer_f32 = np.frombuffer(buffer,np.single)
      tx.do_radae_tx(buffer_f32,tx_out)
      sys.stdout.buffer.write(tx_out)

   eoo_out = np.zeros(tx.Neoo,dtype=np.csingle)
   tx.do_eoo(eoo_out)
   sys.stdout.buffer.write(eoo_out)