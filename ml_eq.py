"""
ML experiment in EQs

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--n_syms', type=int, default=10000, help='number of symbols to train with')
parser.add_argument('--EbNodB', type=float, default=100, help='energy per bit over spectral noise density in dB')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--loss_phase',  action='store_true', help='')
parser.add_argument('--phase_offset',  action='store_true', help='insert random phase offset')
parser.add_argument('--eq', type=str, default='ml', help='equaliser ml/bypass/lin (default ml)')
parser.add_argument('--notrain',  action='store_false', dest='train', help='bypass training (default train, then inference)')
parser.add_argument('--noplots',  action='store_false', dest='plots', help='disable plots (default plots enabled)')
parser.add_argument('--save_model', type=str, default="", help='after training, save model using this filename')
parser.add_argument('--load_model', type=str, default="", help='before inference, load model using this filename')
parser.add_argument('--curve', type=str, default="", help='before inference, load model using this filename')
parser.set_defaults(train=True)
parser.set_defaults(plots=True)
args = parser.parse_args()
n_syms = args.n_syms

bps = 2
batch_size = 16
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

class aQPSKDataset(torch.utils.data.Dataset):
    def __init__(self, n_syms):        
        self.n_syms = n_syms
        self.bits = torch.sign(torch.rand(self.n_syms, bps)-0.5)
        self.symbs = (self.bits[:,::2] + 1j*self.bits[:,1::2])/np.sqrt(2.0)

    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return self.symbs[index,:]
   
# Generalised network for equalisation, we provide n_pilots and n_data symbols,
# hopefully network will determine which pilots EQ which data symbols.  We
# feed the data symbols into each layer so they are available at the end for
# final EQ, DenseNet style.  Hopefully both pilot and data symbol information will be
# used in EQ process.  Give it a few layers to approximate non-linear functions
# like cos,sin and arg[] that are used in classical DSP ML.
class EQ(nn.Module):
    def __init__(self, n_pilots, n_data, EbNodB):
        super().__init__()
        self.n_pilots = n_pilots
        self.n_data = n_data
        self.EbNodB = EbNodB

        self.dense1 = nn.Linear(self.n_pilots+self.n_data, w1)
        self.dense2 = nn.Linear(w1+self.n_data, w1)
        self.dense3 = nn.Linear(w1+self.n_data, w1)
        self.dense4 = nn.Linear(w1+self.n_data, w1)
        self.dense5 = nn.Linear(w1+self.n_data, self.n_data)

    # note complex values passed in as real,imag pairs
    def equaliser(self, pilots, data):
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
        if args.phase_offset:
            phi[:,] = 2*torch.pi*torch.rand(batch_size,1)
        EsNodB = self.EbNodB + 3
        sigma = 10**(-EsNodB/20)
        rx_frame = tx_frame*torch.exp(1j*phi) + sigma*torch.randn_like(tx_frame) 

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
        if args.eq == "bypass":
            rx_data_eq = rx_data
        if args.eq == "ml":
            rx_data_eq = self.equaliser(rx_pilots, rx_data)
        if args.eq == "lin":
            #sum = torch.zeros((batch_size,1), dtype=torch.complex64, device=tx_data.device)
            sum = torch.sum(rx_pilots[:,::2] + 1j*rx_pilots[:,1::2],dim=1)
            phase_est = torch.angle(sum)
            x = rx_frame[:,1]*torch.exp(-1j*phase_est)   
            #print(sum.shape,phase_est.shape,x.shape)
                   
            rx_data_eq = torch.zeros((batch_size,2*n_data), device=tx_data.device)
            rx_data_eq[:,0] = x.real
            rx_data_eq[:,1] = x.imag
            
        # real version of tx_data symbol for loss function
        tx_data_real = torch.zeros((batch_size,2*n_data), device=tx_data.device)
        tx_data_real[:,0] = tx_data[:,0].real
        tx_data_real[:,1] = tx_data[:,0].imag
        tx_data_real = tx_data_real.to(device)

        return tx_data_real,rx_data,rx_data_eq
        
# sym and sym hat real,imag pairs
def loss_phase_mse(sym_hat, sym):
    sym = sym[:,0] + 1j*sym[:,1]
    sym_hat = sym_hat[:,0] + 1j*sym_hat[:,1]
    error = torch.angle(sym*torch.conj(sym_hat))
    loss = torch.sum(error**2)
    return loss

model = EQ(2*n_pilots, 2*n_data, args.EbNodB).to(device)
print(model)
nb_params = sum(p.numel() for p in model.parameters())
print(f" {nb_params} weights")

if args.train:
    if args.loss_phase:
        loss_fn = loss_phase_mse
    else:
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
            loss = loss_fn(rx_data_eq,tx_data_real)
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
        
    if len(args.save_model):
        print(f"Saving model to: {args.save_model}")
        torch.save(model.state_dict(), args.save_model)


# Inference using trained model (or non-ML sim if bypass_eq)
if len(args.load_model):
    print(f"Loading model from: {args.load_model}")
    model.load_state_dict(torch.load(args.load_model,weights_only=True))
model.eval()

def single_point(EbNodB, n_syms):
    bits = torch.sign(torch.rand(n_syms, bps)-0.5)
    tx_data = (bits[:,::2] + 1j*bits[:,1::2])/np.sqrt(2.0)
    model.EbNodB = EbNodB
    with torch.no_grad():
        tx_data = tx_data.to(device)
        tx_data_real,rx_data,rx_data_eq = model(tx_data)
    tx_data_real = tx_data_real.cpu().numpy()
    rx_data = rx_data.cpu().numpy()
    rx_data_eq = rx_data_eq.cpu().numpy()

    n_errors = np.sum(-tx_data_real.flatten()*rx_data_eq.flatten()>0)
    n_bits = n_syms*bps
    BER = n_errors/n_bits
    print(f"EbNodB: {EbNodB:5.2f} n_bits: {n_bits:d} n_errors: {n_errors:d} BER: {BER:5.3f}")

    return BER, rx_data, rx_data_eq

if len(args.curve):
    EbNodB = np.array([0,1,2,3,4,5,6,7,8], dtype=np.float32)
    n_tests = len(EbNodB)
    curve = np.zeros((n_tests,2))
    for i in np.arange(0,n_tests):
        curve[i,0] = EbNodB[i]
        curve[i,1],rx_data,rx_data_eq = single_point(EbNodB[i], args.n_syms)
    np.savetxt(args.curve, curve)
else:
    # single point with scatter plot
    ber,rx_data,rx_data_eq = single_point(args.EbNodB, args.n_syms)
    if args.plots:
        plt.plot(rx_data[:,0],rx_data[:,1],'+')
        plt.plot(rx_data_eq[:,0],rx_data_eq[:,1],'+')
        plt.axis([-2,2,-2,2])
        plt.show()

