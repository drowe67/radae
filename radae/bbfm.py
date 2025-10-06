"""
   Baseband FM version of Radio Autoencoder

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

import torch
from torch import nn
from . import radae_base
import sys
from collections import OrderedDict
import math as m

class BBFM(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim,
                 CNRdB,
                 fd_Hz=5000,
                 fm_Hz=3000,
                 stateful_decoder = False,
                 frames_per_step = 4
                ):

        super(BBFM, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim
        self.CNRdB = CNRdB
        self.fd_Hz = fd_Hz
        self.fm_Hz = fm_Hz
        self.stateful_decoder = stateful_decoder

        # TODO: nn.DataParallel() shouldn't be needed
        self.core_encoder =  nn.DataParallel(radae_base.CoreEncoder(feature_dim, latent_dim, bottleneck=1, frames_per_step=frames_per_step))
        self.core_decoder =  nn.DataParallel(radae_base.CoreDecoder(latent_dim, feature_dim, frames_per_step=frames_per_step))
        self.core_encoder_statefull =  nn.DataParallel(radae_base.CoreEncoderStatefull(feature_dim, latent_dim, bottleneck=1, frames_per_step=frames_per_step))
        self.core_decoder_statefull =  nn.DataParallel(radae_base.CoreDecoderStatefull(latent_dim, feature_dim, frames_per_step=frames_per_step))

        self.enc_stride = frames_per_step
        self.dec_stride = frames_per_step

        if self.dec_stride % self.enc_stride != 0:
            raise ValueError(f"get_decoder_chunks_generic: encoder stride does not divide decoder stride")

        self.Tf = 0.01                                 # feature update period (s) 
        self.Tz = self.Tf*self.enc_stride              # autoencoder latent vector update period (s)
        self.Rz = 1/self.Tz
        self.Rb =  latent_dim/self.Tz                  # payload data BPSK symbol rate (symbols/s or Hz)

        self.beta = self.fd_Hz/self.fm_Hz              # deviation
        self.BWfm = 2*(self.fd_Hz + self.fm_Hz)        # BW estimate using Carsons rule
        self.Gfm = 10*m.log10(3*(self.beta**2)*(self.beta+1))

        print(f"Rb: {self.Rb:5.2f} Deviation: {self.fd_Hz}Hz Max Modn freq: {self.fm_Hz}Hz Beta: {self.beta:3.2f}", file=sys.stderr)

    # Stateful decoder wasn't present during training, so we need to load weights from existing decoder
    def core_decoder_statefull_load_state_dict(self):

        # some of the layer names have been changed due to use of custom GRUStatefull layer
        def key_transformation(old_key):
            for gru in range(1,6):
                if old_key == f"module.gru{gru:d}.weight_ih_l0":
                    return f"module.gru{gru:d}.gru.weight_ih_l0"
                if old_key == f"module.gru{gru:d}.weight_hh_l0":
                    return f"module.gru{gru:d}.gru.weight_hh_l0"
                if old_key == f"module.gru{gru:d}.bias_ih_l0":
                    return f"module.gru{gru:d}.gru.bias_ih_l0"
                if old_key == f"module.gru{gru:d}.bias_hh_l0":
                    return f"module.gru{gru:d}.gru.bias_hh_l0"
            return old_key

        state_dict = self.core_decoder.state_dict()
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key_transformation(key)
            new_state_dict[new_key] = value

        self.core_decoder_statefull.load_state_dict(new_state_dict)
   
    # Stateful encoder wasn't present during training, so we need to load weights from existing encoder
    def core_encoder_statefull_load_state_dict(self):

        # some of the layer names have been changed due to use of custom GRUStatefull layer
        def key_transformation(old_key):
            for gru in range(1,6):
                
                if old_key == f"module.gru{gru:d}.weight_ih_l0":
                    return f"module.gru{gru:d}.gru.weight_ih_l0"
                if old_key == f"module.gru{gru:d}.weight_hh_l0":
                    return f"module.gru{gru:d}.gru.weight_hh_l0"
                if old_key == f"module.gru{gru:d}.bias_ih_l0":
                    return f"module.gru{gru:d}.gru.bias_ih_l0"
                if old_key == f"module.gru{gru:d}.bias_hh_l0":
                    return f"module.gru{gru:d}.gru.bias_hh_l0"
                   
            return old_key

        state_dict = self.core_encoder.state_dict()
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key_transformation(key)
            new_state_dict[new_key] = value

        self.core_encoder_statefull.load_state_dict(new_state_dict)
   
    # stand alone receiver, takes received symbols z and returns features f
    def receiver(self, z_hat):
        if self.stateful_decoder:
            print("stateful!", file=sys.stderr)
            features_hat = torch.empty(1,0,self.feature_dim)
            for i in range(z_hat.shape[1]):
                features_hat = torch.cat([features_hat, self.core_decoder_statefull(z_hat[:,i:i+1,:])],dim=1)
        else:
            features_hat = self.core_decoder(z_hat)
        #print(features_hat.shape, z_hat.shape, file=sys.stderr)
        
        return features_hat
    
    def num_timesteps_at_rate_Rs(self, num_ten_ms_timesteps):
        num_seconds = num_ten_ms_timesteps * self.Tf
        return int(num_seconds*self.Rb)

    def num_10ms_times_steps_rounded_to_modem_frames(self, num_ten_ms_timesteps):
        num_modem_frames = num_ten_ms_timesteps // self.enc_stride
        num_ten_ms_timesteps_rounded = num_modem_frames * self.enc_stride
        #(num_ten_ms_timesteps,  num_modem_frames, num_ten_ms_timesteps_rounded)
        return num_ten_ms_timesteps_rounded

    def forward(self, features, H):
        
        (num_batches, num_ten_ms_timesteps, num_features) = features.shape
        num_timesteps_at_rate_Rs = self.num_timesteps_at_rate_Rs(num_ten_ms_timesteps)
        #print(num_ten_ms_timesteps, num_timesteps_at_rate_Rs)
    
        # For every symbol time step, we need one channel sample
        #print(features.shape,H.shape, features.device, H.device)
        assert (H.shape[0] == num_batches)
        assert (H.shape[1] == num_timesteps_at_rate_Rs)
        assert (H.shape[2] == 1)

        # run encoder, outputs sequence of symbols that each describe 40ms of speech
        z = self.core_encoder(features)
        z_shape = z.shape
        z_hat = torch.reshape(z,(num_batches,num_timesteps_at_rate_Rs,1))
        
        # determine FM demod SNR using piecewise approximation implemented with relus to be torch-friendly
        # note SNR is a vector, 1 sample for symbol as SNR evolves with H
        CNRdB = 20*torch.log10(H) + self.CNRdB
        print(H.shape,CNRdB.shape)
        SNRdB_relu = torch.relu(CNRdB-12) + 12 + self.Gfm
        SNRdB_relu += -torch.relu(-(CNRdB-12))*(1 + self.Gfm/3)
        SNR = 10**(SNRdB_relu/10)

        # note sigma is a vector, noise power evolves across each symbol with H
        sigma = 1/(SNR**0.5)
        n = sigma*torch.randn_like(z_hat)
        z_hat = torch.clamp(z_hat + n, min=-1.0,max=1.0)
                        
        z_hat = torch.reshape(z_hat,z_shape)
        #print(z.shape, z_hat.shape)
        
        features_hat = self.core_decoder(z_hat)
        
        return {
            "features_hat" : features_hat,
            "z_hat"  : z_hat,
            "sigma"  : sigma,
            "CNRdB"  : CNRdB
        }
