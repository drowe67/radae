"""
ML experiment in EQs

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt
import torch.nn.functional as F

class aQPSKDataset(torch.utils.data.Dataset):
    def __init__(self, n_syms):        
        self.bps = 2
        self.n_syms = n_syms
        self.bits = torch.sign(torch.rand(self.n_syms, self.bps)-0.5)
        self.symbs = (self.bits[:,::2] + 1j*self.bits[:,1::2])/np.sqrt(2.0)

    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return self.symbs[index,:]
   
parser = argparse.ArgumentParser()
parser.add_argument('--n_syms', type=int, default=10000, help='number of symbols to train with')
parser.add_argument('--EsNodB', type=float, default=10, help='energy per symbol over spectral noise density in dB')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
args = parser.parse_args()
n_syms = args.n_syms
EsNodB = args.EsNodB

batch_size = 4
w1 = 32
n_pilots = 2
n_data = 1

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Generalised network for equalisation, we provide n_pilots and n_data symbols,
# hopefully network will determine which pilots EQ which data symbols.  We
# feed the data symbols into each layer so they are available at the end for
# final EQ, DenseNet style.  Hopefully pilot and data symbol information will be
# used in EQ process.  Give it a few layers to approximate non-linear functions
# like trig and arg[].
class EQ(nn.Module):
    def __init__(self, n_pilots, n_data):
        super().__init__()
        self.n_pilots = n_pilots
        self.n_data = n_data
        self.dense1 = nn.Linear(self.n_pilots+self.n_data, w1)
        self.dense2 = nn.Linear(w1+self.n_data, w1)
        self.dense3 = nn.Linear(w1+self.n_data, w1)
        self.dense4 = nn.Linear(w1+self.n_data, w1)
        self.dense5 = nn.Linear(w1+self.n_data, self.n_data)

    # note complex values passed in as real,imag pairs
    def equaliser(self, pilots,data):
        x = torch.relu(self.dense1(torch.cat([pilots, data],-1)))
        x = torch.relu(self.dense2(torch.cat([x, data],-1)))
        x = torch.relu(self.dense3(torch.cat([x, data],-1)))
        x = torch.relu(self.dense4(torch.cat([x, data],-1)))
        equalised_data = self.dense5(torch.cat([x, data],-1))
        return equalised_data

    # tx_data is complex, return real values real,imag pairs to suit loss function
    def forward(self, tx_data):
        batch_size = tx_data.shape[0]

        # create frame
        pilot = torch.zeros((batch_size,1),dtype=torch.complex64, device=tx_data.device)
        pilot[:,0] = 1 + 1j*0
        tx_frame = torch.cat([pilot, tx_data, pilot],-1)
        
        # channel simulation, apply same phase offset to all symbols in frame
        phi = torch.zeros(batch_size, tx_frame.shape[1], device=tx_data.device)
        phi[:,] = 2*torch.pi*torch.rand(batch_size,1)
        rx_frame = tx_frame*torch.exp(1j*phi)

        # de-frame pilots and data, separate real and imag
        # TODO: more efficient way to do this
        rx_pilots = torch.zeros((batch_size,2*n_pilots), device=tx_data.device)
        rx_pilots[:,0] = rx_frame[:,0].real
        rx_pilots[:,1] = rx_frame[:,0].imag
        rx_pilots[:,2] = rx_frame[:,2].real
        rx_pilots[:,3] = rx_frame[:,2].imag
        rx_data = torch.zeros((batch_size,2*n_data), device=tx_data.device)
        rx_data[:,0] = rx_frame[:,1].real
        rx_data[:,1] = rx_frame[:,1].imag

        # run equaliser
        rx_data_eq = self.equaliser(rx_pilots, rx_data)

        # real version of tx_data symbol for loss function
        tx_data_real = torch.zeros((batch_size,2*n_data), device=tx_data.device)
        tx_data_real[:,0] = tx_data[:,0].real
        tx_data_real[:,1] = tx_data[:,0].imag
        tx_data_real = tx_data_real.to(device)

        return tx_data_real,rx_data, rx_data_eq
        
model = EQ(2*n_pilots, 2*n_data).to(device)
print(model)
nb_params = sum(p.numel() for p in model.parameters())
print(f" {nb_params} weights")
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

dataset = aQPSKDataset(n_syms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Train model
for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(tx_data) in enumerate(dataloader):
        tx_data = tx_data.to(device)
        tx_data_real,rx_data,rx_data_eq = model(tx_data)
        loss = loss_fn(tx_data_real, rx_data_eq)
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


# Inference using trained model
n_syms_inf = 1000
model.eval()
bits = torch.sign(torch.rand(n_syms_inf, dataset.bps)-0.5)
tx_data = (bits[:,::2] + 1j*bits[:,1::2])/np.sqrt(2.0)
with torch.no_grad():
    tx_data = tx_data.to(device)
    tx_data_real,rx_data,rx_data_eq = model(tx_data)
rx_data = rx_data.cpu().numpy()
rx_data_eq = rx_data_eq.cpu().numpy()
print(rx_data_eq.shape)
plt.plot(rx_data[:,0],rx_data[:,1],'+')
plt.plot(rx_data_eq[:,0],rx_data_eq[:,1],'+')
plt.axis([-2,2,-2,2])
plt.show()

