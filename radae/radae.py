"""
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

""" Pytorch implementations of rate distortion optimized variational autoencoder """

import math as m

import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
from torch.nn.utils import weight_norm

# Quantization and rate related utily functions


def noise_quantize(x):
    """ simulates quantization with addition of random uniform noise """
    return x + (torch.rand_like(x) - 0.5)


# loss functions for vocoder features
def distortion_loss(y_true, y_pred, rate_lambda=None):

    if y_true.size(-1) != 20:
        raise ValueError('distortion loss is designed to work with 20 features')

    ceps_error   = y_pred[..., :18] - y_true[..., :18]
    pitch_error  = 2*(y_pred[..., 18:19] - y_true[..., 18:19])
    corr_error   = y_pred[..., 19:] - y_true[..., 19:]
    pitch_weight = torch.relu(y_true[..., 19:] + 0.5) ** 2

    loss = torch.mean(ceps_error ** 2 + 3. * (10/18) * torch.abs(pitch_error) * pitch_weight + (1/18) * corr_error ** 2, dim=-1)
    loss = torch.mean(loss)
    # reduce bias towards lower Eb/No when training over a range of Eb/No
    #loss = torch.mean(torch.sqrt(torch.mean(loss, dim=1)))

    return loss



# weight initialization and clipping
def init_weights(module):

    if isinstance(module, nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                nn.init.orthogonal_(p[1])


#Simulates 8-bit quantization noise
def n(x):
    return torch.clamp(x + (1./127.)*(torch.rand_like(x)-.5), min=-1., max=1.)


#Wrapper for 1D conv layer
class MyConv(nn.Module):
    def __init__(self, input_dim, output_dim, dilation=1):
        super(MyConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation=dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=2, padding='valid', dilation=dilation)
    def forward(self, x, state=None):
        device = x.device
        conv_in = torch.cat([torch.zeros_like(x[:,0:self.dilation,:], device=device), x], -2).permute(0, 2, 1)
        return torch.tanh(self.conv(conv_in)).permute(0, 2, 1)

#Gated Linear Unit activation
class GLU(nn.Module):
    def __init__(self, feat_size):
        super(GLU, self).__init__()

        torch.manual_seed(5)

        self.gate = weight_norm(nn.Linear(feat_size, feat_size, bias=False))

        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)\
            or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x):

        out = x * torch.sigmoid(self.gate(x))

        return out


#Encoder takes input features and computes symbols to be transmitted
class CoreEncoder(nn.Module):
    STATE_HIDDEN = 128
    FRAMES_PER_STEP = 4
    CONV_KERNEL_SIZE = 4

    def __init__(self, feature_dim, output_dim):

        super(CoreEncoder, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
         
        # derived parameters
        self.input_dim = self.FRAMES_PER_STEP * self.feature_dim

        # Layers are organized like a DenseNet
        self.dense_1 = nn.Linear(self.input_dim, 64)
        self.gru1 = nn.GRU(64, 64, batch_first=True)
        self.conv1 = MyConv(128, 96)
        self.gru2 = nn.GRU(224, 64, batch_first=True)
        self.conv2 = MyConv(288, 96, dilation=2)
        self.gru3 = nn.GRU(384, 64, batch_first=True)
        self.conv3 = MyConv(448, 96, dilation=2)
        self.gru4 = nn.GRU(544, 64, batch_first=True)
        self.conv4 = MyConv(608, 96, dilation=2)
        self.gru5 = nn.GRU(704, 64, batch_first=True)
        self.conv5 = MyConv(768, 96, dilation=2)

        self.z_dense = nn.Linear(864, self.output_dim)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"encoder: {nb_params} weights")

        # initialize weights
        self.apply(init_weights)


    def forward(self, features):

        # Groups FRAMES_PER_STEP frames together in one bunch -- equivalent
        # to a learned transform of size FRAMES_PER_STEP across time. Outputs
        # fewer vectors than the input has because of that
        x = torch.reshape(features, (features.size(0), features.size(1) // self.FRAMES_PER_STEP, self.FRAMES_PER_STEP * features.size(2)))

        batch = x.size(0)
        device = x.device

        # run encoding layer stack
        x = n(torch.tanh(self.dense_1(x)))
        x = torch.cat([x, n(self.gru1(x)[0])], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.gru2(x)[0])], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.gru3(x)[0])], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.gru4(x)[0])], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.gru5(x)[0])], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)
        z = torch.tanh(self.z_dense(x))

        return z



#Decode symbols to reconstruct the vocoder features
class CoreDecoder(nn.Module):

    FRAMES_PER_STEP = 4

    def __init__(self, input_dim, output_dim):
        """ core decoder for RADAE

            Computes features from latents, initial state, and quantization index

        """

        super(CoreDecoder, self).__init__()

        # hyper parameters
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.input_size = self.input_dim

        # Layers are organized like a DenseNet
        self.dense_1    = nn.Linear(self.input_size, 96)
        self.gru1 = nn.GRU(96, 96, batch_first=True)
        self.conv1 = MyConv(192, 32)
        self.gru2 = nn.GRU(224, 96, batch_first=True)
        self.conv2 = MyConv(320, 32)
        self.gru3 = nn.GRU(352, 96, batch_first=True)
        self.conv3 = MyConv(448, 32)
        self.gru4 = nn.GRU(480, 96, batch_first=True)
        self.conv4 = MyConv(576, 32)
        self.gru5 = nn.GRU(608, 96, batch_first=True)
        self.conv5 = MyConv(704, 32)
        self.output  = nn.Linear(736, self.FRAMES_PER_STEP * self.output_dim)
        self.glu1 = GLU(96)
        self.glu2 = GLU(96)
        self.glu3 = GLU(96)
        self.glu4 = GLU(96)
        self.glu5 = GLU(96)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"decoder: {nb_params} weights")
        # initialize weights
        self.apply(init_weights)

    def forward(self, z):

        # run decoding layer stack
        x = n(torch.tanh(self.dense_1(z)))

        x = torch.cat([x, n(self.glu1(n(self.gru1(x)[0])))], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.glu2(n(self.gru2(x)[0])))], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.glu3(n(self.gru3(x)[0])))], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.glu4(n(self.gru4(x)[0])))], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.glu5(n(self.gru5(x)[0])))], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)

        # output layer and reshaping. We produce FRAMES_PER_STEP vocoder feature
        # vectors for every decoded vector of symbols
        x10 = self.output(x)
        features = torch.reshape(x10, (x10.size(0), x10.size(1) * self.FRAMES_PER_STEP, x10.size(2) // self.FRAMES_PER_STEP))

        return features


class RADAE(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim,
                 EbNodB,
                 multipath_delay = 0.002,
                 range_EbNo = False,
                 ber_test = False,
                 rate_Fs = False
                ):

        super(RADAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim
        self.range_EbNo = range_EbNo
        self.ber_test = ber_test
        self.multipath_delay = multipath_delay # Multipath Poor (MPP) path delay (s)
        self.rate_Fs = rate_Fs

        # TODO: nn.DataParallel() shouldn't be needed
        self.core_encoder =  nn.DataParallel(CoreEncoder(feature_dim, latent_dim))
        self.core_decoder =  nn.DataParallel(CoreDecoder(latent_dim, feature_dim))
        #self.core_encoder = CoreEncoder(feature_dim, latent_dim)
        #self.core_decoder = CoreDecoder(latent_dim, feature_dim)

        self.enc_stride = CoreEncoder.FRAMES_PER_STEP
        self.dec_stride = CoreDecoder.FRAMES_PER_STEP

        if self.dec_stride % self.enc_stride != 0:
            raise ValueError(f"get_decoder_chunks_generic: encoder stride does not divide decoder stride")

        self.Tf = 0.01                                 # feature update period (s) 
        self.Tz = self.Tf*self.enc_stride              # autoencoder latent vector update period (s)
        self.Rz = 1/self.Tz
        self.Rb =  latent_dim/self.Tz                  # BPSK symbol rate (symbols/s or Hz)

        # set up noise sigma if doing a fixed Eb/No run (e.g. inference)
        if self.range_EbNo == False:
            self.A = 1                                 # Amplitude of BPSK symbols ~ 1 due to encoder tanh()
            self.Eb = self.A**2                        # Energy of BPSK symbols
            EbNo = 10**(EbNodB/10)                     # linear Eb/No
            self.sigma  = self.A/m.sqrt(EbNo)          # AWGN simulation noise std dev, power spread between real and imag

            # SNR calcs
            B = 3000                                   # bandwidth for measuring noise power (Hz)
            SNR = (EbNo)*(self.Rb/B)
            SNRdB = 10*m.log10(SNR)
            print(f"EbNodB.: {EbNodB:5.2f}  sigma: {self.sigma:5.2f}")
            print(f"SNR3kdB: {SNRdB:5.2f}  Rb...:  {self.Rb:7.2f}")

        # set up OFDM "modem frame" parameters to support multipath simulation.  Modem frame is Nc carriers 
        # wide in frequency and Ns symbols in duration 
        bps = 2                                         # BPSK symbols per QPSK symbol
        Ts = 0.02                                       # OFDM QPSK symbol period
        Rs = 1/Ts                                       # OFDM QPSK symbol rate
        Nsmf = self.latent_dim // bps                   # total number of QPSK symbols in a modem frame across all carriers
        Ns = int(self.Tz // Ts)                         # duration of "modem frame" in QPSK symbols
        Nc = int(Nsmf // Ns)                            # number of carriers
        assert Ns*Nc*bps == latent_dim                  # sanity check, one modem frame should contain all the latent features
        self.Rs = Rs
        self.Ns = Ns
        self.Nc = Nc
        print(f"Nsmf: {Nsmf:3d} Ns: {Ns:3d} Nc: {Nc:3d}")

        # DFT matrices for Nc freq samples, M time samples (could be a FFT but matrix convenient for small, non power of 2 DFTs)
        self.Fs = 8000                                               # sample rate of modem signal 
        self.M = int(self.Fs // self.Rs)                             # oversampling rate
        self.w = 2*m.pi*(400 + torch.arange(Nc)*Rs)/self.Fs          # carrier frequencies, start at 400Hz to be above analog filtering in radios
        self.Winv = torch.zeros((Nc,self.M), dtype=torch.complex64)  # inverse DFT matrix, Nc freq domain to M time domain (OFDM Tx)
        self.Wfwd = torch.zeros((self.M,Nc), dtype=torch.complex64)  # forward DFT matrix, M time domain to Nc freq domain (OFDM Rx)
        for c in range(0,Nc):
           self.Winv[c,:] = torch.exp( 1j*torch.arange(self.M)*self.w[c])/self.M
           self.Wfwd[:,c] = torch.exp(-1j*torch.arange(self.M)*self.w[c])

    def get_sigma(self):
        return self.sigma
    
    def get_Rs(self):
        return self.Rs
    
    def get_Rb(self):
        return self.Rb

    def get_Nc(self):
        return self.Nc
    
    def get_enc_stride(self):
        return self.enc_stride
    
    def get_Ns(self):
        return self.Ns
    
    def forward(self, features, H):
        
        (num_batches, num_ten_ms_timesteps, num_features) = features.shape
        num_timesteps_at_rate_Rs = int((num_ten_ms_timesteps // self.enc_stride)*self.Ns)

        # For every OFDM modem time step, we need one channel sample for each carrier
        print(features.shape,H.shape, features.device, H.device)
        assert (H.shape[0] == num_batches)
        assert (H.shape[1] == num_timesteps_at_rate_Rs)
        assert (H.shape[2] == self.Nc)

        # complex AWGN noise
        if self.range_EbNo:
            EbNodB = -2 + 15*torch.rand(num_batches, 1, 1, device=z.device)
            sigma = 10**(-EbNodB/20)
        else:
            sigma = self.sigma

        # run encoder, outputs sequence of latents that each describe 40ms of speech
        z = self.core_encoder(features)
        if self.ber_test:
            z = torch.sign(torch.rand_like(z)-0.5)
        
        # map z to QPSK symbols, note Es = var(tx_sym) = 2 var(z) = 2 
        # assuming |z| ~ 1 after training
        tx_sym = z[:,:,::2] + 1j*z[:,:,1::2]
        qpsk_shape = tx_sym.shape
        
        # reshape into sequence of OFDM modem frames
        tx_sym = torch.reshape(tx_sym,(num_batches,num_timesteps_at_rate_Rs,self.Nc))

        tx = None
        rx = None
        if self.rate_Fs:
            # Simulate channel at M=Fs/Rs samples per QPSK symbol ---------------------------------

            # IDFT to transform Nc carriers to M time domain samples
            tx = torch.matmul(tx_sym, self.Winv)
            
            # TODO Add cyclic prefix, reshape to (batch,timessteps*M), time domain multipath simulation

            # simulate Power Amplifier (PA) that saturates at abs(tx) ~ 1
            tx_norm = tx/torch.abs(tx)
            tx = torch.tanh(torch.abs(tx)) * tx_norm

            rx = tx + sigma*torch.randn_like(tx)/m.sqrt(self.M)

            # DFT to transform M time domain samples to Nc carriers
            rx_sym = torch.matmul(rx, self.Wfwd)
        else:
            # Simulate channel at one sample per QPSK symbol (Fs=Rs) --------------------------------
            
            # multipath, multiply by per-carrier channel magnitudes at each OFDM modem timestep
            # preserve tx_sym variable so we can return it to measure power after multipath channel
            tx_sym = tx_sym * H

            # note noise power sigma**2 is split between real and imag channels
            n = sigma*torch.randn_like(tx_sym)   
            rx_sym = tx_sym + n

        # demap QPSK symbols
        rx_sym = torch.reshape(rx_sym,qpsk_shape)

        z_hat = torch.zeros_like(z)
        z_hat[:,:,::2] = rx_sym.real
        z_hat[:,:,1::2] = rx_sym.imag

        if self.ber_test:
            n_errors = torch.sum(-z*z_hat>0)
            n_bits = torch.numel(z)
            BER = n_errors/n_bits
            print(f"n_bits: {n_bits:d} BER: {BER:5.3f}")
            
        features_hat = self.core_decoder(z_hat)

        return {
            "features_hat" : features_hat,
            "z_hat"  : z_hat,
            "tx_sym" : tx_sym,
            "tx"     : tx,
            "rx"     : rx
       }
