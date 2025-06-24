"""

From: "NOISE-ROBUST DSP-ASSISTED NEURAL PITCH ESTIMATION
WITH VERY LOW COMPLEXITY" Subramani et al, 2024

Pitch Estimation Models and dataloaders,
    - Classification Based (Input features, output logits)
"""

import torch
import numpy as np

class ftDNNXcorr(torch.nn.Module):

    def __init__(self, input_dim=90, gru_dim=64, output_dim=192):
        super().__init__()

        self.activation = torch.nn.Tanh()

        self.conv = torch.nn.Sequential(
            torch.nn.ZeroPad2d((2, 0, 1, 1)),
            torch.nn.Conv2d(1, 8, 3, bias=True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 8, 3, bias=True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 1, 3, bias=True),
            self.activation,
        )

        self.downsample = torch.nn.Sequential(
            torch.nn.Linear(input_dim, gru_dim),
            self.activation
        )
        self.GRU = torch.nn.GRU(input_size=gru_dim, hidden_size=gru_dim, num_layers=1, batch_first=True)
        self.upsample = torch.nn.Sequential(
            torch.nn.Linear(gru_dim,output_dim),
            self.activation
        )

    def forward(self, x):
        x = self.conv(x.unsqueeze(-1).permute(0,3,2,1)).squeeze(1)
        x,_ = self.GRU(self.downsample(x.permute(0,2,1)))
        x = self.upsample(x).permute(0,2,1)
        logits_softmax = torch.nn.Softmax(dim = 1)(x).permute(0,2,1)

        return logits_softmax

# Dataloader

class ftDNNDataloader(torch.utils.data.Dataset):
      def __init__(self, features, file_pitch, xcorr_dim, sequence_length):
            self.xcorr = np.fromfile(features, dtype=np.float32).reshape(-1,xcorr_dim)
            self.ground_truth = np.fromfile(file_pitch, dtype=np.float32)
            self.sequence_length = sequence_length

            frame_max = self.xcorr.shape[0]//sequence_length
            #print(self.xcorr.shape, sequence_length, frame_max)
            
            self.xcorr = np.reshape(self.xcorr[:frame_max*sequence_length,:], (frame_max, sequence_length, xcorr_dim))
            self.ground_truth = np.reshape(self.ground_truth[:frame_max*sequence_length], (frame_max, sequence_length))
            # TODO refactor to rm confidence 
            #print(self.xcorr.shape, self.ground_truth.shape)

      def __len__(self):
            return self.xcorr.shape[0]

      def __getitem__(self, index):
            return torch.from_numpy(self.xcorr[index,:,:]),torch.from_numpy(self.ground_truth[index])
