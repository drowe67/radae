"""
ML experiment in training OFDM pilots

usage: python3 ml_pilots.py

Pilots as complex numbers -> IDFT -> tanh -> AWGN -> detector -> loss function
                                                        ^
                                                        |
                                                      pilots

+ could start with current pilots of randoms
+ tanh encourages low PAPR
+ as well as AWGN could we use existing model17 to train against?
+ what does loss function do?  Maximises SNR of detected signal?  What are
  we comparing to?  Dtmax at correct point?  Probability of detection?
+ with power and PAPR constrained by tanh, maximising Dt over a range of noise
  values would be useful
+ make pilots trainable
+ what other properties can we optimise for? Well maybe multipath channels, or
  freq offsets, or levels.  But lets start with something simple, mainly classical 
  DSP
+ detector could be a NN, or trainable
+ there is no dataset, which is interesting

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt
from radae import RADAE

parser = argparse.ArgumentParser()
parser.add_argument('--n_syms', type=int, default=10000, help='number of symbols to train with')
parser.add_argument('--EsNodB', type=float, default=10, help='energy per symbol over spectral noise desnity in dB')
parser.add_argument('--epochs', type=int, default=10, help='number of trarining epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
args = parser.parse_args()
n_syms = args.n_syms
EsNodB = args.EsNodB

batch_size = 32

# Get cpu, gpu or mps device for training.
device = "cpu"
print(f"Using {device} device")

class aDataset(torch.utils.data.Dataset):
    def __init__(self,
                Nc,       
                n_syms):

        self.Nc = Nc
        self.n_syms = n_syms
        
    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return torch.zeros(self.Nc*2)

    # OFDM pilots, one complex number per carrier, represented as two float biases
# These are actually trainable constants, we just use the biases, inputs set to 0.
class Pilots(nn.Module):
    def __init__(self, Rb, Nc, M, Fs, Winv, EsNodB):
        super().__init__()
        self.Rb = Rb
        self.Nc = Nc
        self.M = M
        self.Fs = Fs
        self.Winv = Winv
        self.P = nn.Linear(Nc*2, Nc*2)
        # assume signal power will be about 1 due to tanh
        S = 1
        EsNo = 10 ** (EsNodB/10)
        self.sigma = (S*self.Fs/(EsNo*self.Rb))**(0.5)

    def forward(self, input):
        # IDFT to a M point time domain sequence
        P = self.P(input);
        P = P[:,::2] + 1j*P[:,1::2]
        p = torch.matmul(P*self.M/(Nc**0.5),self.Winv)
        tx = p
        # limit power of complex sequence
        tx = torch.tanh(torch.abs(tx))*torch.exp(1j*torch.angle(tx))
        # AWGN channel
        rx = tx + self.sigma*torch.randn_like(tx)
        # detector
        Dt = torch.sum(torch.conj(rx)*tx,dim=1)/(self.Nc*self.M)
        return torch.abs(Dt),P

def my_loss(Dt,P):
    loss1 = -torch.sum(Dt)                # maximise correlation peak
    loss2 = 0.1*torch.std(torch.abs(P))   # encourage pilot symbol power to be constant
    return loss1+loss2

# Bring up a RADAE model to obtain various constants
latent_dim = 40
num_features = 20
num_used_features = 20
r = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004)
Rb = r.Rb; Nc = r.Nc; M = r.M; Fs = r.Fs; Winv = r.Winv
model = Pilots(Rb,Nc,M,Fs,Winv,EsNodB).to(device)
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

dataset = aDataset(Nc, n_syms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Train model
for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(x) in enumerate(dataloader):
        Dt,P = model(x)
        # we want to maximise correlation, so invert loss
        loss = my_loss(Dt,P)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if np.isnan(loss.item()):
            print("NAN encountered - quitting (try reducing lr)!")
            quit()
        sum_loss += loss.item()

    print(f'Epochs:{epoch + 1:5d} | ' \
        f'Batches per epoch: {batch + 1:3d} | ' \
        f'Loss: {sum_loss / (batch + 1):.10f}')

for layer in model.children():
    if isinstance(layer, nn.Linear):
        P = layer.state_dict()['bias']

P = P[::2] + 1j*P[1::2];
p = torch.matmul(P,r.Winv)
p = p.cpu().detach().numpy()
P = P.cpu().detach().numpy()
print(P)
PAPRdB = 20*np.log10(np.max(np.abs(p))/np.mean(np.abs(p)))
print(f"PAPR: {PAPRdB:f}")
fig, ax = plt.subplots(2, 1,figsize=(6,6))
ax[0].set_title('P complex plane')
ax[0].plot(P.real, P.imag,'b+')
ax[0].axis([-4,4,-4,4])
ax[1].plot(np.abs(p))
ax[1].set_title('|p|')
plt.show(block=False)
plt.pause(0.001)
input("hit[enter] to end.")
plt.close('all')

