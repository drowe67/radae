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

""" Pytorch implementations of rate distortion optimized variational autoencoder """

import math as m

import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
from torch.nn.utils.parametrizations import weight_norm
from matplotlib import pyplot as plt

# Quantization and loss utility functions

def noise_quantize(x):
    """ simulates quantization with addition of random uniform noise """
    return x + (torch.rand_like(x) - 0.5)


# loss functions for vocoder features
def distortion_loss(y_true, y_pred):

    if y_true.size(-1) != 20:
        raise ValueError('distortion loss is designed to work with 20 features')

    ceps_error   = y_pred[..., :18] - y_true[..., :18]
    pitch_error  = 2*(y_pred[..., 18:19] - y_true[..., 18:19])
    corr_error   = y_pred[..., 19:] - y_true[..., 19:]
    pitch_weight = torch.relu(y_true[..., 19:] + 0.5) ** 2

    loss = torch.mean(ceps_error ** 2 + 3. * (10/18) * torch.abs(pitch_error) * pitch_weight + (1/18) * corr_error ** 2, dim=-1)
    loss = torch.mean(loss, dim=-1)

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

# Generate pilots using Barker codes which have good correlation properties
def barker_pilots(Nc):
    P_barker_8  = torch.tensor([1., 1., 1., -1., -1., 1., -1.])
    P_barker_13 = torch.tensor([1., 1., 1., 1., 1., -1., -1., 1., 1., -1., 1., -1., 1])

    # repeating length 8 Barker code 
    P = torch.zeros(Nc,dtype=torch.complex64)
    for i in range(Nc):
        P[i] = P_barker_13[i % len(P_barker_13)]
    return P

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

    def __init__(self, feature_dim, output_dim, papr_opt=False):

        super(CoreEncoder, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
        self.papr_opt           = papr_opt

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

        if self.papr_opt:
            # power unconstrained here, constrained in time domain forward() instead
            z = self.z_dense(x)
        else:
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
                 rate_Fs = False,
                 papr_opt = False,
                 phase_offset = 0,
                 freq_offset = 0,
                 df_dt = 0,
                 gain = 1,
                 freq_rand = False,
                 gain_rand = False,
                 pilots = False,
                 pilot_eq = False,
                 eq_mean6 = True,
                 cyclic_prefix = 0,
                 time_offset = 0,
                 coarse_mag = False
                ):

        super(RADAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim
        self.EbNodB = EbNodB
        self.range_EbNo = range_EbNo
        self.ber_test = ber_test
        self.multipath_delay = multipath_delay 
        self.rate_Fs = rate_Fs
        self.papr_opt = papr_opt
        self.phase_offset = phase_offset
        self.freq_offset = freq_offset
        self.df_dt = df_dt
        self.gain = gain
        self.freq_rand = freq_rand
        self.gain_rand = gain_rand
        self.pilots = pilots
        self.pilot_eq = pilot_eq
        self.per_carrier_eq = True
        self.phase_mag_eq = False
        self.eq_mean6 = eq_mean6
        self.time_offset = time_offset
        self.coarse_mag = coarse_mag

        # TODO: nn.DataParallel() shouldn't be needed
        self.core_encoder =  nn.DataParallel(CoreEncoder(feature_dim, latent_dim, papr_opt=papr_opt))
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
        self.Rb =  latent_dim/self.Tz                  # payload data BPSK symbol rate (symbols/s or Hz)

        # set up OFDM "modem frame" parameters to support multipath simulation.  Modem frame is Nc carriers 
        # wide in frequency and Ns symbols in duration 
        bps = 2                                         # BPSK symbols per QPSK symbol

        if self.pilots:
            Ts = 0.03                                   # OFDM QPSK symbol period (without pilots or CP)
        else:
            Ts = 0.02
        Rs = 1/Ts                                       # OFDM QPSK symbol rate
        Nzmf = 3                                        # number of latent vectors in a modem frame
        Nsmf = Nzmf*self.latent_dim // bps              # total number of QPSK symbols in a modem frame across all carriers
        
        Ns = int(Nzmf*self.Tz / Ts)                    # duration of "modem frame" in QPSK symbols
        print(self.Tz, Ts, Nzmf*self.Tz / Ts, Ns)
        #quit()
        Tmf = Ns*Ts                                     # period of modem frame (s), this must remain constant for real time operation
        Nc = int(Nsmf // Ns)                            # number of carriers
        assert Ns*Nc*bps == Nzmf*latent_dim             # sanity check, one modem frame should contain all the latent features
        
        # when inserting pilots increase OFDM symbol rate so that modem frame period is constant
        Rs_dash = Rs
        Ts_dash = Ts
        Rb_dash = self.Rb
        
        if self.pilots:
            Rs_dash = Rs*(Ns+1)/Ns
            Ts_dash = 1/Rs_dash
            Rb_dash = self.Rb*(Ns+1)/Ns
        
        # when inserting cyclic prefix increase OFDM symbol rate so that modem frame period is constant
        self.Fs = 8000                                               # sample rate of modem signal 
        self.d_samples = int(self.multipath_delay * self.Fs)         # multipath delay in samples
        self.Ncp = int(cyclic_prefix*self.Fs)
        
        Rs_dash = Rs_dash/(1-cyclic_prefix/Ts_dash)            
        Rb_dash = Rb_dash/(1-cyclic_prefix/Ts_dash)
        Ts_dash = 1/Rs_dash
        
        # DFT matrices for Nc freq samples, M time samples (could be a FFT but matrix convenient for small, non power of 2 DFTs)
        self.M = round(self.Fs / Rs_dash)                            # oversampling rate
        lower = round(400/Rs_dash)                                   # start carrier freqs at about 400Hz to be above analog filtering in radios
        self.w = 2*m.pi*(lower+torch.arange(Nc))/self.M              # note: must be integer DFT freq indexes or DFT falls over
        self.Winv = torch.zeros((Nc,self.M), dtype=torch.complex64)  # inverse DFT matrix, Nc freq domain to M time domain (OFDM Tx)
        self.Wfwd = torch.zeros((self.M,Nc), dtype=torch.complex64)  # forward DFT matrix, M time domain to Nc freq domain (OFDM Rx)
        for c in range(0,Nc):
           self.Winv[c,:] = torch.exp( 1j*torch.arange(self.M)*self.w[c])/self.M
           self.Wfwd[:,c] = torch.exp(-1j*torch.arange(self.M)*self.w[c])
        
        self.P = (2**(0.5))*barker_pilots(Nc)
        self.p = torch.matmul(self.P,self.Winv)

        self.d_samples = int(self.multipath_delay * self.Fs)         # multipath delay in samples
        self.Ncp = int(cyclic_prefix*self.Fs)
    
        print(f"Rs: {Rs:5.2f} Rs': {Rs_dash:5.2f} Ts': {Ts_dash:5.3f} Nsmf: {Nsmf:3d} Ns: {Ns:3d} Nc: {Nc:3d} M: {self.M:d} Ncp: {self.Ncp:d}")

        self.Tmf = Tmf
        self.bps = bps
        self.Ts = Ts
        self.Ts_dash = Ts_dash
        self.Rb_dash = Rb_dash
        self.Rs = Rs
        self.Rs_dash = Rs_dash
        self.Ns = Ns
        self.Nc = Nc
        self.Nzmf = Nzmf

    
    def move_device(self, device):
        # TODO: work out why we need this step
        self.Winv = self.Winv.to(device)
        self.Wfwd = self.Wfwd.to(device)
 
    def num_timesteps_at_rate_Rs(self, num_ten_ms_timesteps):
        num_modem_frames = num_ten_ms_timesteps / self.enc_stride / self.Nzmf
        return int(num_modem_frames*self.Ns)
    
    def num_timesteps_at_rate_Fs(self, num_timesteps_at_rate_Rs):
        if self.pilots:
            Ns = self.Ns
            return int(((Ns+1)/Ns)*num_timesteps_at_rate_Rs*(self.M+self.Ncp))
        else:
            return int(num_timesteps_at_rate_Rs*self.M)
        
    def num_10ms_times_steps_rounded_to_modem_frames(self, num_ten_ms_timesteps):
        num_modem_frames = num_ten_ms_timesteps // self.enc_stride // self.Nzmf
        num_ten_ms_timesteps_rounded = num_modem_frames * self.enc_stride * self.Nzmf
        #(num_ten_ms_timesteps,  num_modem_frames, num_ten_ms_timesteps_rounded)
        return num_ten_ms_timesteps_rounded
    
    # Use classical DSP pilot based equalisation. Note just for inference atm
    def do_pilot_eq(self, num_modem_frames, rx_sym_pilots):
        Nc = self.Nc 

        # First, estimate the (complex) value of each received pilot symbol
        rx_pilots = torch.zeros(num_modem_frames, Nc, dtype=torch.complex64)
        if self.per_carrier_eq:
            # estimate pilot symbol for each carrier by smoothing information from adjacent pilots; moderate loss, but
            # handles multipath and timing offsets
            for i in torch.arange(num_modem_frames):
                if self.eq_mean6:
                    #  3-pilot local mean across frequency
                    rx_pilots[i,0] = torch.mean(rx_sym_pilots[0,i,0,0:3]/self.P[0:3])
                    #rx_pilots[i,0] = rx_sym_pilots[0,i,0,0]/self.P[0]
                    for c in torch.arange(1,Nc-1):
                        rx_pilots[i,c] = torch.mean(rx_sym_pilots[0,i,0,c-1:c+2]/self.P[c-1:c+2])
                    rx_pilots[i,Nc-1] = torch.mean(rx_sym_pilots[0,i,0,Nc-3:Nc]/self.P[Nc-3:Nc])
                    #rx_pilots[i,Nc-1] = rx_sym_pilots[0,i,0,Nc-1]/self.P[Nc-1]
                else:
                    #  3-pilot least squares fit across frequency
                    for c in range(Nc):
                        c_mid = c
                        # handle edges, alternative is extra "wingman" pilots
                        if c == 0:
                            c_mid = 1
                        if c == Nc-1:
                            c_mid = Nc-2
                        local_path_delay_s = 0.0025      # guess at actual path delay
                        a = local_path_delay_s*self.Fs
                        A = torch.tensor([[1, torch.exp(-1j*self.w[c_mid-1]*a)], [1, torch.exp(-1j*self.w[c_mid]*a)], [1, torch.exp(-1j*self.w[c_mid+1]*a)]])
                        P = torch.matmul(torch.inverse(torch.matmul(torch.transpose(A,0,1),A)),torch.transpose(A,0,1))
                        h = torch.reshape(rx_sym_pilots[0,i,0,c_mid-1:c_mid+2]/self.P[c_mid-1:c_mid+2],(3,1))
                        g = torch.matmul(P,h)
                        rx_pilots[i,c] = g[0] + g[1]*torch.exp(-1j*self.w[c]*a)
                 
        else:
            # average all pilots together. Low loss, but won't handle multipath and is sensitive to timing offsets
            for i in torch.arange(num_modem_frames):
                rx_pilots[i,:] = torch.mean(rx_sym_pilots[0,i,0,:]/self.P)

        # Linearly interpolate between two pilots to EQ data symbols (phase and optionally mag)
        for i in torch.arange(num_modem_frames-1):
            for c in torch.arange(0,Nc):
                slope = (rx_pilots[i+1,c] - rx_pilots[i,c])/(self.Ns+1)
                # assume pilots at index 0 and Ns+1, we want to linearly interpolate channel at 1...Ns 
                rx_ch = slope*torch.arange(0,self.Ns+2) + rx_pilots[i,c]
                if self.phase_mag_eq:
                    rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]/rx_ch[1:self.Ns+1]
                else:
                    rx_ch_angle = torch.angle(rx_ch)
                    rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]*torch.exp(-1j*rx_ch_angle[1:self.Ns+1])
        # last modem frame, use previous slope
        i = num_modem_frames-1
        for c in torch.arange(0,Nc):
            rx_ch = slope*torch.arange(0,self.Ns+2) + rx_pilots[i,c]
            if self.phase_mag_eq:
                rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]/rx_ch[1:self.Ns+1]
            else:
                rx_ch_angle = torch.angle(rx_ch)
                rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]*torch.exp(-1j*rx_ch_angle[1:self.Ns+1])

        # Optional "coarse" magntiude estimation and correction based on mean of all pilots across sequence. Unlike 
        # regular PSK, ML network is sensitive to magnitude shifts.  We can't use the average mangnitude of the non-pilot symbols
        # as they have unknown amplitudes. TODO: For a practical, real world implementation, make this a frame by frame AGC type
        # algorithm, e.g. IIR smoothing of the RMS mag of each frames pilots 
        if self.coarse_mag:
            # est RMS magnitude
            mag = torch.mean(torch.abs(rx_pilots)**2)**0.5
            #print(mag)
            rx_sym_pilots = rx_sym_pilots/mag

        return rx_sym_pilots
    
    # rate Fs receiver
    def receiver(self, rx):
        Ns = self.Ns
        if self.pilots:
            Ns = Ns + 1
        # integer number of modem frames
        num_timesteps_at_rate_Rs = len(rx) // (self.M+self.Ncp)
        num_modem_frames = num_timesteps_at_rate_Rs // Ns
        num_timesteps_at_rate_Rs = Ns * num_modem_frames
        rx = rx[:num_timesteps_at_rate_Rs*(self.M+self.Ncp)]

        # remove cyclic prefix
        rx = torch.reshape(rx,(1,num_timesteps_at_rate_Rs,self.M+self.Ncp))
        rx_dash = rx[:,:,self.Ncp+self.time_offset:self.Ncp+self.time_offset+self.M]
        
        # DFT to transform M time domain samples to Nc carriers
        rx_sym = torch.matmul(rx_dash, self.Wfwd)
        
        if self.pilots:
            rx_sym_pilots = torch.reshape(rx_sym,(1, num_modem_frames, self.Ns+1, self.Nc))
            if self.pilot_eq:
                rx_sym_pilots = self.do_pilot_eq(num_modem_frames,rx_sym_pilots)
            rx_sym = torch.ones(1, num_modem_frames, self.Ns, self.Nc, dtype=torch.complex64)
            rx_sym = rx_sym_pilots[:,:,1:self.Ns+1,:]

        # demap QPSK symbols
        rx_sym = torch.reshape(rx_sym, (1, -1, self.latent_dim//2))
        z_hat = torch.zeros(1,rx_sym.shape[1], self.latent_dim)
        #print(rx_sym.shape,z_hat.shape, z_hat.device)
        
        z_hat[:,:,::2] = rx_sym.real
        z_hat[:,:,1::2] = rx_sym.imag
            
        features_hat = self.core_decoder(z_hat)
        
        return features_hat,z_hat


    def forward(self, features, H, G=None):
        
        (num_batches, num_ten_ms_timesteps, num_features) = features.shape
        num_timesteps_at_rate_Rs = self.num_timesteps_at_rate_Rs(num_ten_ms_timesteps)
        #print(num_ten_ms_timesteps, num_timesteps_at_rate_Rs)

        # For every OFDM modem time step, we need one channel sample for each carrier
        #print(features.shape,H.shape, features.device, H.device)
        assert (H.shape[0] == num_batches)
        assert (H.shape[1] == num_timesteps_at_rate_Rs)
        assert (H.shape[2] == self.Nc)

        # AWGN noise
        if self.range_EbNo:
            EbNodB = -6 + 20*torch.rand(num_batches,1,1,device=features.device)
        else:           
            EbNodB = torch.tensor(self.EbNodB)

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
   
        # optionally insert pilot symbols, at the start of each modem frame
        if self.pilots:
            num_modem_frames = num_timesteps_at_rate_Rs // self.Ns
            tx_sym = torch.reshape(tx_sym,(num_batches, num_modem_frames, self.Ns, self.Nc))
            tx_sym_pilots = torch.zeros(num_batches, num_modem_frames, self.Ns+1, self.Nc, dtype=torch.complex64,device=tx_sym.device)
            tx_sym_pilots[:,:,1:self.Ns+1,:] = tx_sym
            tx_sym_pilots[:,:,0,:] = self.P
            num_timesteps_at_rate_Rs = num_timesteps_at_rate_Rs + num_modem_frames
            tx_sym = torch.reshape(tx_sym_pilots,(num_batches, num_timesteps_at_rate_Rs, self.Nc))

        tx = None
        rx = None
        if self.rate_Fs:
            num_timesteps_at_rate_Fs = num_timesteps_at_rate_Rs*self.M
 
            # Simulate channel at M=Fs/Rs samples per QPSK symbol ---------------------------------

            # IDFT to transform Nc carriers to M time domain samples
            tx = torch.matmul(tx_sym, self.Winv)

            # Optionally insert a cyclic prefix
            Ncp = self.Ncp
            if self.Ncp:
                tx_cp = torch.zeros((num_batches,num_timesteps_at_rate_Rs,self.M+Ncp),dtype=torch.complex64,device=tx.device)
                tx_cp[:,:,Ncp:] = tx
                tx_cp[:,:,:Ncp] = tx_cp[:,:,-Ncp:]
                tx = tx_cp
                num_timesteps_at_rate_Fs = num_timesteps_at_rate_Rs*(self.M+Ncp)
            tx = torch.reshape(tx,(num_batches,num_timesteps_at_rate_Fs))                         
            
            # simulate Power Amplifier (PA) that saturates at abs(tx) ~ 1
            # TODO: this has problems when magnitude goes thru 0 (e.g. --ber_test), change formulation to use angle()
            if self.papr_opt:
                tx_norm = tx/torch.abs(tx)
                tx = torch.tanh(torch.abs(tx)) * tx_norm
            
            # rate Fs multipath model
            d = self.d_samples
            tx_mp = torch.zeros((num_batches,num_timesteps_at_rate_Fs))
            tx_mp = tx*G[:,:,0]
            tx_mp[:,d:] = tx_mp[:,d:] + tx[:,:-d]*G[:,:-d,1]
            # normalise power through multipath model
            tx_power = torch.mean(torch.abs(tx)**2)
            tx_mp_power = torch.mean(torch.abs(tx_mp)**2)
            mp_gain = (tx_power/tx_mp_power)**0.5
            tx = mp_gain*tx_mp
            
            # user supplied phase and freq offsets (used at inference time)
            if self.phase_offset:
                phase = self.phase_offset*torch.ones_like(tx)
                phase = torch.exp(1j*phase)
                tx = tx*phase
            if self.freq_offset:
                freq = torch.zeros(num_batches, num_timesteps_at_rate_Fs)
                freq[:,] = self.freq_offset*torch.ones(num_timesteps_at_rate_Fs) + self.df_dt*torch.arange(num_timesteps_at_rate_Fs)/self.Fs
                omega = freq*2*torch.pi/self.Fs
                lin_phase = torch.cumsum(omega,dim=1)
                lin_phase = torch.exp(1j*lin_phase)
                tx = tx*lin_phase

            # insert per sequence random phase and freq offset (training time)
            if self.freq_rand:
                phase = torch.zeros(num_batches, num_timesteps_at_rate_Fs,device=tx.device)
                phase[:,] = 2.0*torch.pi*torch.rand(num_batches,1)
                # TODO maybe this should be +/- Rs/2
                freq_offset = 40*(torch.rand(num_batches,1) - 0.5)
                omega = freq_offset*2*torch.pi/self.Fs
                lin_phase = torch.zeros(num_batches, num_timesteps_at_rate_Fs,device=tx.device)
                lin_phase[:,] = omega*torch.arange(num_timesteps_at_rate_Fs)
                tx = tx*torch.exp(1j*(phase+lin_phase))
            
            # AWGN noise
            EbNodB = torch.reshape(EbNodB,(num_batches,1))
            EbNo = 10**(EbNodB/10)
            
            if self.papr_opt:
                # determine sigma assuming rms power var(tx) = 1 (will be a few dB less due to PA backoff)
                S = 1
                sigma = (S*self.Fs/(EbNo*self.Rb))**(0.5)
            else:
                # similar to rate Rs, but scale noise by M samples/symbol
                sigma = (EbNo*(self.M))**(-0.5)
            rx = tx + sigma*torch.randn_like(tx)

            # insert per sequence random gain variations, -20 ... +20 dB (training time)
            if self.gain_rand:
                gain = torch.zeros(num_batches, num_timesteps_at_rate_Fs,device=tx.device)
                gain[:,] = -20 + 40*torch.rand(num_batches,1)
                #print(gain[0,:3])
                gain = 10 ** (gain/20)
                rx = rx * gain

            # user supplied gain    
            rx = rx * self.gain

            # remove cyclic prefix (assume genie timing)
            rx = torch.reshape(rx,(num_batches,num_timesteps_at_rate_Rs,self.M+self.Ncp))
            rx_dash = rx[:,:,Ncp+self.time_offset:Ncp+self.time_offset+self.M]

            # DFT to transform M time domain samples to Nc carriers
            rx_sym = torch.matmul(rx_dash, self.Wfwd)
        else:
            # Simulate channel at one sample per QPSK symbol (Fs=Rs) --------------------------------
            
            if self.phase_offset:
                phase = self.phase_offset*torch.ones_like(tx_sym)
                phase = torch.exp(1j*phase)
                tx_sym = tx_sym*phase

            # multipath, multiply by per-carrier channel magnitudes at each OFDM modem timestep
            # preserve tx_sym variable so we can return it to measure power after multipath channel
            tx_sym = tx_sym * H

            # note noise power sigma**2 is split between real and imag channels
            sigma = 10**(-EbNodB/20)
            n = sigma*torch.randn_like(tx_sym)
            rx_sym = tx_sym + n
            
        # strip out the pilots if present (TODO pass to ML decoder network, lots of useful information)
        if self.pilots:
            rx_sym_pilots = torch.reshape(rx_sym,(num_batches, num_modem_frames, self.Ns+1, self.Nc))

            if self.pilot_eq:
                rx_sym_pilots = self.do_pilot_eq(num_modem_frames,rx_sym_pilots)

            rx_sym = torch.ones(num_batches, num_modem_frames, self.Ns, self.Nc, dtype=torch.complex64)
            rx_sym = rx_sym_pilots[:,:,1:self.Ns+1,:]

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
            "rx"     : rx,
            "sigma"  : sigma.cpu().numpy(),
            "EbNodB" : EbNodB.cpu().numpy()
       }
