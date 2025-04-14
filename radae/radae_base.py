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

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from collections import OrderedDict
import sys

# Quantization and loss utility functions

def noise_quantize(x):
    """ simulates quantization with addition of random uniform noise """
    return x + (torch.rand_like(x) - 0.5)


# loss functions for vocoder features
def distortion_loss(y_true, y_pred):

    if y_true.size(-1) != 20 and y_true.size(-1) != 21:
        raise ValueError('distortion loss is designed to work with 20 or 21 features')

    ceps_error   = y_pred[..., :18] - y_true[..., :18]
    pitch_error  = 2*(y_pred[..., 18:19] - y_true[..., 18:19])
    corr_error   = y_pred[..., 19:20] - y_true[..., 19:20]
    pitch_weight = torch.relu(y_true[..., 19:20] + 0.5) ** 2
    data_error = 0
    if y_true.size(-1) == 21:
        data_error = y_pred[..., 20:21] - y_true[..., 20:21]
    loss = torch.mean(ceps_error ** 2 + 3. * (10/18) * torch.abs(pitch_error) * pitch_weight + (1/18) * corr_error ** 2 + (0.5/18)*data_error ** 2, dim=-1)
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

# Wrapper for GRU layer that maintains state internally
class GRUStatefull(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_first):
        super(GRUStatefull, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.states = torch.zeros(1,1,self.hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=batch_first)
    def forward(self, x):
        gru_out,self.states = self.gru(x,self.states)
        return gru_out
    def reset(self):
       self.states = torch.zeros(1,1,self.hidden_dim)

# Wrapper for conv1D layer that maintains state internally
class Conv1DStatefull(nn.Module):
    def __init__(self, input_dim, output_dim, dilation=1):
        super(Conv1DStatefull, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.kernel_size = 2
        self.states_len = self.kernel_size-1+self.dilation-1
        self.states = torch.zeros(1,self.states_len,self.input_dim)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=self.kernel_size, padding='valid', dilation=dilation)

    def forward(self, x):
        conv_in = torch.cat([self.states,x],dim=1)
        self.states = conv_in[:,-self.states_len:,:]
        conv_in = conv_in.permute(0, 2, 1)
        return torch.tanh(self.conv(conv_in)).permute(0, 2, 1)
    
    def reset(self):
        self.states = torch.zeros(1,self.states_len,self.input_dim)

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

    def __init__(self, feature_dim, output_dim, bottleneck = 1, frames_per_step=4):

        super(CoreEncoder, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
        self.bottleneck         = bottleneck
        self.frames_per_step   = frames_per_step

        # derived parameters
        self.input_dim = self. frames_per_step * self.feature_dim

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
        print(f"encoder: {nb_params} weights", file=sys.stderr)

        # initialize weights
        self.apply(init_weights)


    def forward(self, features):

        # Groups FRAMES_PER_STEP frames together in one bunch -- equivalent
        # to a learned transform of size FRAMES_PER_STEP across time. Outputs
        # fewer vectors than the input has because of that
        x = torch.reshape(features, (features.size(0), features.size(1) // self.frames_per_step, self.frames_per_step * features.size(2)))

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

        # bottleneck constrains 1D real symbol magnitude
        if self.bottleneck == 1:
            z = torch.tanh(self.z_dense(x))
        else:
            z = self.z_dense(x)

        return z

# Stateful version of Encoder
class CoreEncoderStatefull(nn.Module):

    def __init__(self, feature_dim, output_dim, bottleneck = 1, frames_per_step = 4):

        super(CoreEncoderStatefull, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
        self.bottleneck         = bottleneck
        self.frames_per_step   = frames_per_step
       
        # derived parameters
        self.input_dim = self.frames_per_step * self.feature_dim

        # Layers are organized like a DenseNet
        self.dense_1 = nn.Linear(self.input_dim, 64)
        self.gru1 = GRUStatefull(64, 64, batch_first=True)
        self.conv1 = Conv1DStatefull(128, 96)
        self.gru2 = GRUStatefull(224, 64, batch_first=True)
        self.conv2 = Conv1DStatefull(288, 96, dilation=2)
        self.gru3 = GRUStatefull(384, 64, batch_first=True)
        self.conv3 = Conv1DStatefull(448, 96, dilation=2)
        self.gru4 = GRUStatefull(544, 64, batch_first=True)
        self.conv4 = Conv1DStatefull(608, 96, dilation=2)
        self.gru5 = GRUStatefull(704, 64, batch_first=True)
        self.conv5 = Conv1DStatefull(768, 96, dilation=2)

        self.z_dense = nn.Linear(864, self.output_dim)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"encoder: {nb_params} weights", file=sys.stderr)

        # initialize weights
        self.apply(init_weights)


    def forward(self, features):

        # Groups FRAMES_PER_STEP frames together in one bunch -- equivalent
        # to a learned transform of size FRAMES_PER_STEP across time. Outputs
        # fewer vectors than the input has because of that
        x = torch.reshape(features, (features.size(0), features.size(1) // self.frames_per_step, self.frames_per_step * features.size(2)))

        # run encoding layer stack
        x = n(torch.tanh(self.dense_1(x)))
        x = torch.cat([x, n(self.gru1(x))], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.gru2(x))], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.gru3(x))], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.gru4(x))], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.gru5(x))], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)

        # bottleneck constrains 1D real symbol magnitude
        if self.bottleneck == 1:
            z = torch.tanh(self.z_dense(x))
        else:
            z = self.z_dense(x)

        return z



#Decode symbols to reconstruct the vocoder features
class CoreDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, frames_per_step = 4):
        """ core decoder for RADAE

            Computes features from latents, initial state, and quantization index

        """

        super(CoreDecoder, self).__init__()

        # hyper parameters
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.input_size = self.input_dim
        self.frames_per_step = frames_per_step

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
        self.output  = nn.Linear(736, self.frames_per_step * self.output_dim)
        self.glu1 = GLU(96)
        self.glu2 = GLU(96)
        self.glu3 = GLU(96)
        self.glu4 = GLU(96)
        self.glu5 = GLU(96)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"decoder: {nb_params} weights", file=sys.stderr)
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
        features = torch.reshape(x10, (x10.size(0), x10.size(1) * self.frames_per_step, x10.size(2) // self.frames_per_step))

        return features

# Decode symbols to reconstruct the vocoder features. Statefull version that can processes sequences 
# as short as one z vector, and maintains it's own internal state
class CoreDecoderStatefull(nn.Module):

    def __init__(self, input_dim, output_dim, frames_per_step = 4):
        """ core decoder for RADAE

            Computes features from latent z

        """

        super(CoreDecoderStatefull, self).__init__()

        # hyper parameters
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.input_size = self.input_dim
        self.frames_per_step = frames_per_step

        # Layers are organized like a DenseNet
        self.dense_1    = nn.Linear(self.input_size, 96)
        self.gru1 = GRUStatefull(96, 96, batch_first=True)
        self.conv1 = Conv1DStatefull(192, 32)
        self.gru2 = GRUStatefull(224, 96, batch_first=True)
        self.conv2 = Conv1DStatefull(320, 32)
        self.gru3 = GRUStatefull(352, 96, batch_first=True)
        self.conv3 = Conv1DStatefull(448, 32)
        self.gru4 = GRUStatefull(480, 96, batch_first=True)
        self.conv4 = Conv1DStatefull(576, 32)
        self.gru5 = GRUStatefull(608, 96, batch_first=True)
        self.conv5 = Conv1DStatefull(704, 32)
        self.output  = nn.Linear(736, self.frames_per_step * self.output_dim)
        self.glu1 = GLU(96)
        self.glu2 = GLU(96)
        self.glu3 = GLU(96)
        self.glu4 = GLU(96)
        self.glu5 = GLU(96)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"decoder: {nb_params} weights", file=sys.stderr)
        # initialize weights
        self.apply(init_weights)

    def forward(self, z):

        x = n(torch.tanh(self.dense_1(z)))
        x = torch.cat([x, n(self.glu1(n(self.gru1(x))))], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.glu2(n(self.gru2(x))))], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.glu3(n(self.gru3(x))))], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.glu4(n(self.gru4(x))))], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.glu5(n(self.gru5(x))))], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)
        x = self.output(x)

        features = torch.reshape(x,(1,z.shape[1]*self.frames_per_step,self.output_dim))
        return features

    def reset(self):
        self.conv1.reset()
        self.conv2.reset()
        self.conv3.reset()
        self.conv4.reset()
        self.conv5.reset()

        self.gru1.reset()
        self.gru1.reset()
        self.gru2.reset()
        self.gru3.reset()
        self.gru4.reset()
        self.gru5.reset()
        