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
                sequence_length,      # number of feature vectors in each sequence of time steps we train on
                mp_sequence_length,   # corresponding number of multipath vectors in each sequence of time steps we train on
                Nc,
                num_used_features=20,
                num_features=36,
                h_file=""
                ):

        self.sequence_length = sequence_length

        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.features = self.features[:, :num_used_features]
        self.num_sequences = self.features.shape[0] // sequence_length
        
        # optionally load multipath model
        self.mp_sequence_length = mp_sequence_length
        if len(h_file):
            self.H = np.reshape(np.fromfile(h_file, dtype=np.float32), (-1, Nc))
            mp_num_sequences = self.H.shape[0] // mp_sequence_length
            if mp_num_sequences < self.num_sequences:
                print(f"Multipath file too short num_sequences: {self.num_sequences:d} mp_num_sequences: {mp_num_sequences:d}")
                quit()
        else:
            self.H = np.ones((self.num_sequences*self.mp_sequence_length,Nc))
        print(f"dataloader: sequence_length: {self.sequence_length:d} num_sequences: {self.num_sequences:d} mp_sequence_length: {mp_sequence_length:d}")
        print(self.features.shape, self.H.shape)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        H = self.H[index * self.mp_sequence_length: (index + 1) * self.mp_sequence_length, :]

        return features,H
