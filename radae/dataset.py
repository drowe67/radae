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

import torch
import numpy as np

class RADAEDataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,      # number of vocoder feature vectors in each sequence of time steps we train on
                H_sequence_length,    # number of rate Rs multipath channel vectors in each sequence of time steps we train on
                Nc,
                G_sequence_length,    # number of rate Fs multipath channel vectors in each sequence of time steps we train on
                num_used_features=20,
                num_features=36,
                h_file="",            # rate Rs multipath channel samples
                g_file="",            # rate Fs multipath channel samples
                rate_Fs = False
                ):

        self.sequence_length = sequence_length

        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.features = self.features[:, :num_used_features]
        self.num_sequences = self.features.shape[0] // sequence_length
        self.rate_Fs = rate_Fs

        # optionally set up rate Rs multipath model
        self.H_sequence_length = H_sequence_length
        if len(h_file):
            self.H = np.reshape(np.fromfile(h_file, dtype=np.float32), (-1, Nc))
            self.H_num_sequences = self.H.shape[0] // self.H_sequence_length
            if self.H_num_sequences < self.num_sequences:
                print(f"dataloader: Number sequences in multipath H file less than feature file:")
                print(f"dataloader:   num_sequences: {self.num_sequences:d} H_num_sequences: {self.H_num_sequences:d}")
                print(f"dataloader:   If H is large enough to represent the range of channels this is probably OK, we'll re-use H sequences as we train .....")
        else:
            # dummy multipath model that is equivalent to AWGN
            self.H_num_sequences = 100
            self.H = np.ones((self.H_num_sequences*self.H_sequence_length,Nc))

        # optionally set up rate Fs multipath model
        self.G_sequence_length = G_sequence_length
        self.G_num_sequences = 0
        if len(g_file):
            self.G = np.reshape(np.fromfile(g_file, dtype=np.csingle), (-1, 2))
            # first row is hf gain factor
            mp_gain = np.real(self.G[0,0])
            self.G = mp_gain*self.G[1:,:]
            self.G_num_sequences = self.G.shape[0] // self.G_sequence_length
            if self.H_num_sequences < self.num_sequences:
                print(f"dataloader: Number sequences in multipath G file less than feature file:")
                print(f"dataloader:   num_sequences: {self.num_sequences:d} G_num_sequences: {self.G_num_sequences:d}")
                print(f"dataloader:   If G is large enough to represent the range of channels this is probably OK, we'll re-use G sequences as we train .....")
        if len(g_file) == 0 and self.rate_Fs:
            # no mulipath sample file provided, but we are still running at rate Fs, so create a benign (AWGN) model
            self.G_num_sequences = 100
            self.G = np.zeros((self.G_num_sequences*self.G_sequence_length,2), dtype=np.csingle)
            self.G[:,0] = 1
        
        # summary of datasets loaded
        print(f"dataloader: sequence_length..: {self.sequence_length:d} num_sequences: {self.num_sequences:d} features.shape: {self.features.shape}")
        print(f"dataloader: H_sequence_length: {self.H_sequence_length:d} H_num_sequences: {self.H_num_sequences:d} H.shape: {self.H.shape}")
        if self.G_num_sequences > 0:
            print(f"dataloader: G_sequence_length: {self.G_sequence_length:d} G_num_sequences: {self.G_num_sequences:d} G.shape: {self.G.shape}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]

        # deal with H_num_sequences < num_sequences, by re-using H sequences
        h_index = index % (self.H_num_sequences - 1)
        H = self.H[h_index * self.H_sequence_length: (h_index + 1) * self.H_sequence_length, :]

        if self.G_num_sequences > 0:
            # deal with G_num_sequences < num_sequences, by re-using G sequences
            g_index = index % (self.G_num_sequences - 1)
            G = self.G[g_index * self.G_sequence_length: (g_index + 1) * self.G_sequence_length, :]
        else:
            # If G not used (e.g. rate Rs), passing small dummy G doubles training speed
            G = np.zeros((1,2))

        return features,H,G
