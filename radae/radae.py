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
from collections import OrderedDict
from . import radae_base
from . import dsp

# Generate pilots using Barker codes which have good correlation properties
def barker_pilots(Nc):
    P_barker_8  = torch.tensor([1., 1., 1., -1., -1., 1., -1.])
    P_barker_13 = torch.tensor([1., 1., 1., 1., 1., -1., -1., 1., 1., -1., 1., -1., 1])

    # repeating length 8 Barker code 
    P = torch.zeros(Nc,dtype=torch.complex64)
    for i in range(Nc):
        P[i] = P_barker_13[i % len(P_barker_13)]
    return P


class RADAE(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim,
                 EbNodB,
                 multipath_delay = 0.002,
                 range_EbNo = False,
                 range_EbNo_start = -6.0,
                 ber_test = False,
                 rate_Fs = False,
                 bottleneck = 1,
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
                 coarse_mag = False,
                 correct_freq_offset = False,
                 stateful_decoder = False,
                 txbpf_en = False,
                 pilots2 = False,
                 timing_rand = False,
                 correct_time_offset = False,
                 tanh_clipper = False,
                 frames_per_step = 4
                ):

        super(RADAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim
        self.EbNodB = EbNodB
        self.range_EbNo = range_EbNo
        self.range_EbNo_start = range_EbNo_start
        self.ber_test = ber_test
        self.multipath_delay = multipath_delay 
        self.rate_Fs = rate_Fs
        assert bottleneck == 1 or bottleneck == 2 or bottleneck == 3
        self.bottleneck = bottleneck
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
        self.correct_freq_offset = correct_freq_offset
        self.stateful_decoder = stateful_decoder
        self.txbpf_en = txbpf_en
        self.pilots2 = pilots2
        self.timing_rand = timing_rand
        self.correct_time_offset = correct_time_offset
        self.tanh_clipper = tanh_clipper

        # TODO: nn.DataParallel() shouldn't be needed
        self.core_encoder =  nn.DataParallel(radae_base.CoreEncoder(feature_dim, latent_dim, bottleneck=bottleneck, frames_per_step=frames_per_step))
        self.core_decoder =  nn.DataParallel(radae_base.CoreDecoder(latent_dim, feature_dim, frames_per_step=frames_per_step))
        self.core_encoder_statefull =  nn.DataParallel(radae_base.CoreEncoderStatefull(feature_dim, latent_dim, bottleneck=bottleneck, frames_per_step=frames_per_step))
        self.core_decoder_statefull =  nn.DataParallel(radae_base.CoreDecoderStatefull(latent_dim, feature_dim, frames_per_step=frames_per_step))
        #self.core_encoder = CoreEncoder(feature_dim, latent_dim)
        #self.core_decoder = CoreDecoder(latent_dim, feature_dim)

        self.enc_stride = frames_per_step
        self.dec_stride = frames_per_step

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
        
        Ns = int(Nzmf*self.Tz / Ts)                     # duration of "modem frame" in QPSK symbols
        
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
        carrier_1_freq = 1500-Rs_dash*Nc/2                           # centre signal on 1500 Hz offset from carrier (centre of SSB radio passband)
        carrier_1_index = round(carrier_1_freq/Rs_dash)              # DFT index of first carrier, must be an integer for OFDM to work
        self.w = 2*m.pi*(carrier_1_index+torch.arange(Nc))/self.M    # note: must be integer DFT freq indexes or DFT falls over
        self.Winv = torch.zeros((Nc,self.M), dtype=torch.complex64)  # inverse DFT matrix, Nc freq domain to M time domain (OFDM Tx)
        self.Wfwd = torch.zeros((self.M,Nc), dtype=torch.complex64)  # forward DFT matrix, M time domain to Nc freq domain (OFDM Rx)
        for c in range(0,Nc):
           self.Winv[c,:] = torch.exp( 1j*torch.arange(self.M)*self.w[c])/self.M
           self.Wfwd[:,c] = torch.exp(-1j*torch.arange(self.M)*self.w[c])
        
        # set up pilots in freq and time domain
        self.P = (2**(0.5))*barker_pilots(Nc)
        self.p = torch.matmul(self.P,self.Winv)
        self.Pend = torch.clone(self.P)
        self.Pend[1::2] = -1*self.Pend[1::2]
        self.pend = torch.matmul(self.Pend,self.Winv)
        #print(self.P,self.Pend, file=sys.stderr)
        if self.Ncp:
            self.p_cp = torch.zeros(self.Ncp+self.M,dtype=torch.complex64)
            self.p_cp[self.Ncp:] = self.p
            self.p_cp[:self.Ncp] = self.p[-self.Ncp:]
            self.pend_cp = torch.zeros(self.Ncp+self.M,dtype=torch.complex64)
            self.pend_cp[self.Ncp:] = self.pend
            self.pend_cp[:self.Ncp] = self.pend[-self.Ncp:]
        self.pilot_gain = 1.00
        if self.bottleneck == 3:
            pilot_backoff = 10**(-2/20)
            # TODO: I think this expression should have abs(P[0]) in it, see also coarse_mag
            self.pilot_gain = pilot_backoff*self.M/(Nc**0.5)

        self.d_samples = int(self.multipath_delay * self.Fs)         # multipath delay in samples
        self.Ncp = int(cyclic_prefix*self.Fs)

        # set up End Of Over sequence
        # Normal frame ...PDDDDP... 
        # EOO frame    ...PE000E... 
        # Key: P = self.p_cp, D = data symbols, E = self.pend_cp, 0 = zeros
        if self.Ncp:
            M = self.M
            Ncp = self.Ncp
            Nmf = int((Ns+1)*(M+Ncp))
            eoo = torch.zeros(1,Nmf+M+Ncp,dtype=torch.complex64)
            eoo[0,:M+Ncp] = self.p_cp
            eoo[0,M+Ncp:2*(M+Ncp)] = self.pend_cp
            eoo[0,Nmf:Nmf+(M+Ncp)] = self.pend_cp
            eoo *= self.pilot_gain
            if self.bottleneck == 3:
                eoo = torch.tanh(torch.abs(eoo)) * torch.exp(1j*torch.angle(eoo))
            self.eoo = eoo
        
        print(f"frames_per_step: {frames_per_step:d} Tz: {self.Tz:5.3f} Rs: {Rs:5.2f} Rs': {Rs_dash:5.2f} Ts': {Ts_dash:5.3f} Nsmf: {Nsmf:3d} Ns: {Ns:3d} Nc: {Nc:3d} M: {self.M:d} Ncp: {self.Ncp:d}", file=sys.stderr)

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

        if txbpf_en:
            Ntap=51
            bandwidth = 1.2*(self.w[Nc-1] - self.w[0])*self.Fs/(2*torch.pi)
            centre = (self.w[Nc-1] + self.w[0])*self.Fs/(2*torch.pi)/2
            print(f"Tx BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
            txbpf = dsp.complex_bpf(Ntap, self.Fs, bandwidth,centre)
            self.txbpf_conv = nn.Conv1d(1, 1, kernel_size=len(txbpf.h), dtype=torch.complex64)
            self.alpha = txbpf.alpha
            self.txbpf_delay = int(Ntap // 2)
            with torch.no_grad():
                print(self.txbpf_conv.weight.shape)
                self.txbpf_conv.weight[0,0,:] = nn.Parameter(torch.from_numpy(txbpf.h))
                print(self.txbpf_conv.weight[0,0,:])
                self.txbpf_conv.bias = nn.Parameter(torch.zeros(1,dtype=torch.complex64))
                self.txbpf_conv.weight.requires_grad = False
                self.txbpf_conv.bias.requires_grad = False
                
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
   
    def move_device(self, device):
        # TODO: work out why we need this step
        self.w = self.w.to(device)
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
            return int(num_timesteps_at_rate_Rs*(self.M+self.Ncp))
        
    def num_10ms_times_steps_rounded_to_modem_frames(self, num_ten_ms_timesteps):
        num_modem_frames = num_ten_ms_timesteps // self.enc_stride // self.Nzmf
        num_ten_ms_timesteps_rounded = num_modem_frames * self.enc_stride * self.Nzmf
        #(num_ten_ms_timesteps,  num_modem_frames, num_ten_ms_timesteps_rounded)
        return num_ten_ms_timesteps_rounded
    
    # Use classical DSP pilot based equalisation. Note just for inference atm
    # TODO consider moving to dsp.py, or perhaps another file, to reduce the size of this file. 
    # Down side is it has a lot of flags and options that would need passing
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

        # Optional "coarse" magnitude estimation and correction based on mean of all pilots across sequence. Unlike 
        # regular PSK, ML network is sensitive to magnitude shifts.  We can't use the average magnitude of the non-pilot symbols
        # as they have unknown amplitudes. TODO: For a practical, real world implementation, make this a frame by frame AGC type
        # algorithm, e.g. IIR smoothing of the RMS mag of each frames pilots 
        if self.coarse_mag:
            # est RMS magnitude
            mag = torch.mean(torch.abs(rx_pilots)**2)**0.5
            if self.bottleneck == 3:
                mag = mag*torch.abs(self.P[0])/self.pilot_gain
            print(f"coarse mag: {mag:f}", file=sys.stderr)
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
        
        if self.stateful_decoder:
            print("stateful!", file=sys.stderr)
            features_hat = torch.empty(1,0,self.feature_dim)
            for i in range(z_hat.shape[1]):
                features_hat = torch.cat([features_hat, self.core_decoder_statefull(z_hat[:,i:i+1,:])],dim=1)
        else:
            features_hat = self.core_decoder(z_hat)
        print(features_hat.shape,z_hat.shape, file=sys.stderr)
        
        return features_hat,z_hat
    
    # Estimate SNR given a vector r of M received pilot samples
    # rate_Fs/time domain, only works on 1D vectors (i.e. can broadcast or do multiple estimates)
    # unfortunately this doesn't work for multipath channels (good results for AWGN)
    def est_snr(self, r, time_offset=0):
        st = self.Ncp+time_offset
        en = st + self.M
        p = self.p_cp[st:en]
        Ct = torch.abs(torch.dot(torch.conj(r),p))**2 / torch.dot(torch.conj(r),r)
        SNR_est = Ct/(torch.dot(torch.conj(p),p) - Ct)
        return SNR_est.real
    
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
            EbNodB = self.range_EbNo_start + 20*torch.rand(num_batches,1,1,device=features.device)
        else:           
            EbNodB = self.EbNodB*torch.ones(num_batches,1,1,device=features.device)

        # run encoder, outputs sequence of latents that each describe 40ms of speech
        z = self.core_encoder(features)
        if self.ber_test:
            z = torch.sign(torch.rand_like(z)-0.5)
        
        # map z to QPSK symbols, note Es = var(tx_sym) = 2 var(z) = 2 
        # assuming |z| ~ 1 after training
        tx_sym = z[:,:,::2] + 1j*z[:,:,1::2]
        qpsk_shape = tx_sym.shape

        # replace some elements of z with fixed pilots
        if self.pilots2:
            tx_sym[:,:,6::self.Ns] = 0.5*self.pilot_gain*(2**0.5)
            #print(self.pilot_gain,self.P)
            #print(tx_sym.shape)
            #quit()

        # constrain magnitude of 2D complex symbols 
        if self.bottleneck == 2:
            tx_sym = torch.tanh(torch.abs(tx_sym))*torch.exp(1j*torch.angle(tx_sym))
            
        # reshape into sequence of OFDM modem frames
        tx_sym = torch.reshape(tx_sym,(num_batches,num_timesteps_at_rate_Rs,self.Nc))
   
        # optionally insert pilot symbols, at the start of each modem frame
        if self.pilots:
            num_modem_frames = num_timesteps_at_rate_Rs // self.Ns
            tx_sym = torch.reshape(tx_sym,(num_batches, num_modem_frames, self.Ns, self.Nc))
            tx_sym_pilots = torch.zeros(num_batches, num_modem_frames, self.Ns+1, self.Nc, dtype=torch.complex64,device=tx_sym.device)
            tx_sym_pilots[:,:,1:self.Ns+1,:] = tx_sym
            tx_sym_pilots[:,:,0,:] = self.pilot_gain*self.P
            num_timesteps_at_rate_Rs = num_timesteps_at_rate_Rs + num_modem_frames
            tx_sym = torch.reshape(tx_sym_pilots,(num_batches, num_timesteps_at_rate_Rs, self.Nc))

        tx_before_channel = None
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
            
            # Constrain magnitude of complex rate Fs time domain signal, simulates Power
            # Amplifier (PA) that saturates at abs(tx) ~ 1
            if self.bottleneck == 3:
                if self.tanh_clipper:
                    tx = torch.tanh(torch.abs(tx))*torch.exp(1j*torch.angle(tx))
                else:
                    tx = torch.exp(1j*torch.angle(tx))
            tx_before_channel = tx

            # rate Fs multipath model
            d = self.d_samples
            tx_mp = torch.zeros((num_batches,num_timesteps_at_rate_Fs))
            #print(tx.shape, G.shape)
            tx_mp = tx*G[:,:,0]
            tx_mp[:,d:] = tx_mp[:,d:] + tx[:,:-d]*G[:,:-d,1]
            # normalise power through multipath model (used at inference so SNR is correct)
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
            
            if self.bottleneck == 3:
                # determine sigma assuming rms power var(tx) = 1 (actually a fraction of a dB less in practice)
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
            rx_dash = torch.clone(rx)
            
            # inference time correction of freq offset, allows us to produce a rx.f32 file
            # with a freq offset while decoding correctly here
            if self.freq_offset and self.correct_freq_offset:
                rx_dash = rx_dash*torch.conj(lin_phase)
                
            # remove cyclic prefix
            rx_dash = torch.reshape(rx_dash,(num_batches,num_timesteps_at_rate_Rs,self.M+self.Ncp))
            rx_dash = rx_dash[:,:,Ncp+self.time_offset:Ncp+self.time_offset+self.M]
            
            # DFT to transform M time domain samples to Nc carriers
            rx_sym = torch.matmul(rx_dash, self.Wfwd)
        else:
            # Simulate channel at one sample per QPSK symbol (Fs=Rs) --------------------------------

            if self.bottleneck == 3:
                # Hybrid time & freq domain model - we need time domain to apply bottleneck
                # IDFT to transform Nc carriers to M time domain samples
                tx = torch.matmul(tx_sym, self.Winv)
                # Apply time domain magnitude bottleneck - an infinite clipper
                tx = torch.exp(1j*torch.angle(tx))

                # apply BPF-clip stages to obtain a reasonable 99% power bandwidth at low PAPR.  BPF is implemented by
                # shifting signal to baseband an applying a real LPF of bandwidth B/2.  Three stages gives us a 99% power BW
                # of around 1200-1400 Hz at 0 PAPR, loss appears similar to previous waveforms.
                if self.txbpf_en:
                    tx = torch.reshape(tx,(num_batches, 1, num_timesteps_at_rate_Rs*self.M))
                    phase_vec = torch.exp(-1j*self.alpha*torch.arange(0,tx.shape[2],device=tx.device))
                    tx = tx*phase_vec

                    tx = torch.concat((torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device),tx,torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device)),dim=2)
                    tx = self.txbpf_conv(tx)
                    tx = torch.exp(1j*torch.angle(tx))

                    tx = torch.concat((torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device),tx,torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device)),dim=2)
                    tx = self.txbpf_conv(tx)
                    tx = torch.exp(1j*torch.angle(tx))
                    
                    tx = torch.concat((torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device),tx,torch.zeros((num_batches,1,self.txbpf_delay),device=tx.device)),dim=2)
                    tx = self.txbpf_conv(tx)
                    tx = torch.exp(1j*torch.angle(tx))
                    
                    tx = tx*torch.conj(phase_vec)
                    tx = torch.reshape(tx,(num_batches, num_timesteps_at_rate_Rs, self.M))
    
                tx_before_channel = tx
                # DFT to transform M time domain samples to Nc carriers
                tx_sym = torch.matmul(tx, self.Wfwd)
                    
            if self.phase_offset:
                phase = self.phase_offset*torch.ones_like(tx_sym)
                phase = torch.exp(1j*phase)
                tx_sym = tx_sym*phase

            # per-sequence random [-1,1] ms time shift, which results in a linear phase shift across frequency
            # models fine timing errors
            if self.timing_rand:
                d = 0.001*(1 - 2*torch.rand((num_batches,1),device=tx_sym.device))
                # Use vector multiply to create a shape (batch,Nc) 2D tensor
                phase_offset = -d*torch.reshape(self.w,(1,self.Nc))*self.Fs
                phase_offset = torch.reshape(phase_offset,(num_batches,self.Nc,1))
            
                # change to (batch,Nc,timestep), as all time steps get the same phase offset
                tx_sym = tx_sym.permute(0,2,1)
                tx_sym = tx_sym * torch.exp(1j*phase_offset)
                tx_sym = tx_sym.permute(0,2,1)

            # per sequence random [-2,2] fine frequency offset        
            if self.freq_rand:
                freq_offset = 4*torch.rand((num_batches,1),device=tx_sym.device) - 2.0
                omega = freq_offset*2*torch.pi/self.Rs
                # shape (num_batchs,num_timsteps)
                phase = torch.zeros((num_batches,num_timesteps_at_rate_Rs),device=tx_sym.device)
                # broadcast omega across timesteps
                phase[:,] = omega
                # integrate to get phase
                phase = torch.cumsum(phase,dim=1)
                phase = torch.reshape(phase,(num_batches,num_timesteps_at_rate_Rs,1))
                # same freq offset/phase for each carrier
                tx_sym = tx_sym*torch.exp(1j*phase)

            # multipath, multiply by per-carrier channel magnitudes (and optionally phase) at each OFDM modem timestep
            # preserve tx_sym variable so we can return it to measure power after multipath channel
            tx_sym = tx_sym * H

            # AWGN noise ------------------
            # note noise power sigma**2 is split between real and imag channels
            if self.bottleneck == 3:
                EbNo = 10**(EbNodB/10)
                sigma = self.M/((2*self.Nc*EbNo)**(0.5))
                sigma = sigma/(2**0.5)
            else:
                sigma = 10**(-EbNodB/20)
            n = sigma*torch.randn_like(tx_sym)
            rx_sym = tx_sym + n
            
        # strip out the pilots if present (future work: pass to ML decoder network, lots of useful information)
        if self.pilots:
            rx_sym_pilots = torch.reshape(rx_sym,(num_batches, num_modem_frames, self.Ns+1, self.Nc))

            if self.pilot_eq:
                rx_sym_pilots = self.do_pilot_eq(num_modem_frames,rx_sym_pilots)
                
            rx_sym = torch.ones(num_batches, num_modem_frames, self.Ns, self.Nc, dtype=torch.complex64)
            rx_sym = torch.reshape(rx_sym_pilots[:,:,1:self.Ns+1,:],(num_batches, num_modem_frames*self.Ns, self.Nc))

        # genie based phase adjustment for time shift 
        if self.correct_time_offset:
            #print(self.correct_time_offset)
            # Use vector multiply to create a shape (batch,Nc) 2D tensor
            phase_offset = -self.correct_time_offset*torch.reshape(self.w,(1,self.Nc))
            phase_offset = torch.reshape(phase_offset,(num_batches,self.Nc,1))
        
            # change to (batch,Nc,timestep), as all time steps get the same phase offset
            rx_sym = rx_sym.permute(0,2,1)
            rx_sym = rx_sym * torch.exp(1j*phase_offset)
            rx_sym = rx_sym.permute(0,2,1)

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
            "tx"     : tx_before_channel,
            "rx"     : rx,
            "sigma"  : sigma.cpu().numpy(),
            "EbNodB" : EbNodB.cpu().numpy()
       }
