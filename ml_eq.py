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
parser.add_argument('--eq', type=str, default='ml', help='equaliser ml/bypass/dsp (default ml)')
parser.add_argument('--notrain',  action='store_false', dest='train', help='bypass training (default train, then inference)')
parser.add_argument('--noplots',  action='store_false', dest='plots', help='disable plots (default plots enabled)')
parser.add_argument('--save_model', type=str, default="", help='after training, save model using this filename')
parser.add_argument('--load_model', type=str, default="", help='before inference, load model using this filename')
parser.add_argument('--curve', type=str, default="", help='before inference, load model using this filename')
parser.add_argument('--framer', type=int, default=1, help='framer design')
parser.set_defaults(train=True)
parser.set_defaults(plots=True)
args = parser.parse_args()
n_syms = args.n_syms

bps = 2
batch_size = 16
w1 = 32

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class framer:
    def __init__(self):
        self.Nc = 0          
        self.n_data = 0      
        self.n_pilot = 0     # total number of pilot symbols
    def frame(self, pilot, data):
        return None
    def deframe(self, pilot, data):
        return None
    def channel(self, EbNodB, phase_offset):
        return None
    
# single carrier "PDP" frame to get us started
class frame1(framer):
    def __init__(self):
        self.Nc = 1          # num carriers
        self.n_pilot = 2     # num pilot symbols
        self.n_data = 1      # num data symbols

    # symbols arranged in frames as (batch,Nc,timesteps) (timesteps = total # symbols in frame along time axis)
    def frame(self, tx_data):
        batch_size = tx_data.shape[0]
        tx_frame = torch.zeros((batch_size,self.Nc,self.n_pilot+self.n_data),dtype=torch.complex64, device=tx_data.device)
        tx_frame[:,0,0] = 1
        tx_frame[:,0,1] = tx_data[:,0]
        tx_frame[:,0,2] = 1
        return tx_frame
    
    def channel(self, tx_frame, EbNodB):     
        batch_size = tx_frame.shape[0]
        # apply same phase offset to all symbols in frame
        phi = torch.zeros_like(tx_frame, device=tx_frame.device)
        if args.phase_offset:
            phi[:,0,] = 2*torch.pi*torch.rand(batch_size,1)
        EsNodB = EbNodB + 3
        sigma = 10**(-EsNodB/20)
        rx_frame = tx_frame*torch.exp(1j*phi) + sigma*torch.randn_like(tx_frame)
        # extract just data symbols after channel model
        rx_data = torch.reshape(rx_frame[:,0,1],(batch_size,self.n_data))
        return rx_frame, rx_data

    def dsp_equaliser(self, rx_frame):
        batch_size = rx_frame.shape[0]
        sum = rx_frame[:,0,0] + rx_frame[:,0,2]
        phase_est = torch.angle(sum)
        x = rx_frame[:,0,1]*torch.exp(-1j*phase_est)              
        rx_data_eq = torch.zeros((batch_size,2*self.n_data), device=rx_frame.device)
        rx_data_eq[:,0] = x.real
        rx_data_eq[:,1] = x.imag
        return rx_data_eq

# multi-carrier:
# PDDDDP
# PDDDDP
# PDDDDP
    
class frame2(framer):
    def __init__(self):
        self.Nc = 3        # num carriers
        self.n_pilot = 6   # num pilot symbol
        self.n_data = 12   # num data symbols
        self.Ns = 6        # num timesteps in frame (measured in symbols, inc pilots)

    def frame(self, tx_data):
        batch_size = tx_data.shape[0]
        tx_frame = torch.zeros((batch_size,self.Nc,self.Ns),dtype=torch.complex64, device=tx_data.device)
        tx_frame[:,:,0] = 1
        tx_frame[:,:,self.Ns-1] = 1
        for c in np.arange(self.Nc):
            #print(tx_frame[:,c,1:self.Ns-1].shape, tx_data[:,c*(self.Ns-2):(c+1)*(self.Ns-2)].shape)
            tx_frame[:,c,1:self.Ns-1] = tx_data[:,c*(self.Ns-2):(c+1)*(self.Ns-2)]
        #print(tx_data[0,:])
        #print(tx_frame[0,:,])
        #quit()
        return tx_frame
    
    def channel(self, tx_frame, EbNodB):     
        batch_size = tx_frame.shape[0]
        # apply same phase offset to all symbols in frame
        phi = torch.zeros((batch_size, self.Nc, self.Ns), device=tx_frame.device)
        if args.phase_offset:
            phi[:,:,:] = 2*torch.pi*torch.rand((batch_size,1,1), device=tx_frame.device)
        EsNodB = EbNodB + 3
        sigma = 10**(-EsNodB/20)
        #print(tx_frame.shape, phi.shape)
        #print(phi[0,:,:])

        rx_frame = tx_frame*torch.exp(1j*phi) + sigma*torch.randn_like(tx_frame)
        # extract just data symbols in shape (batch,n_data) after channel model
        rx_data = torch.zeros((batch_size, self.n_data), dtype=torch.complex64)
        tmp = rx_frame[:,:,1:self.Ns-1]
        #print(tmp.shape, rx_data.shape)
        
        rx_data = torch.reshape(tmp,(batch_size,self.n_data))
        return rx_frame, rx_data

    def dsp_equaliser(self, rx_frame):
        batch_size = rx_frame.shape[0]
        sum = torch.sum(rx_frame[:,:,0],dim=1) + torch.sum(rx_frame[:,:,self.Ns-1],dim=1)
        #print(sum.shape)
        #print(sum)
        
        phase_est = torch.reshape(torch.angle(sum),(batch_size,1,1))
        tmp = rx_frame*torch.exp(-1j*phase_est)
        tmp = torch.reshape(tmp[:,:,1:self.Ns-1],(batch_size,self.n_data))
        #print(tmp.shape)
        rx_data_eq = torch.zeros((batch_size,2*self.n_data), device=rx_frame.device)
        rx_data_eq[:,::2] = tmp.real
        rx_data_eq[:,1::2] = tmp.imag
        return rx_data_eq

class aQPSKDataset(torch.utils.data.Dataset):
    def __init__(self, n_syms, n_data):        
        self.n_syms = n_syms
        self.n_data = n_data
        self.bits = torch.sign(torch.rand(self.n_syms*self.n_data, bps)-0.5)
        self.symbs = (self.bits[:,::2] + 1j*self.bits[:,1::2])/np.sqrt(2.0)
        self.symbs = torch.reshape(self.symbs, (n_syms,n_data))

    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return self.symbs[index,:]


# helper function to convert 2D complex tensor to (real,imag) float pairs
def tofloat(x):
    x_float = torch.zeros((x.shape[0],2*x.shape[1]), device=x.device)
    x_float[:,::2] = x.real
    x_float[:,1::2] = x.imag
    return x_float

# Generalised network for equalisation, we provide n_pilot and n_data symbols,
# as input, and it returns n_data equalised symbols as output.  Each symbol
# is represented by two floats (the real and imag part).
#
# Hopefully network will determine which pilots EQ which data symbols.  We
# feed the symbols into each layer so (in particular the data symbols) are 
# available at the final layer for EQ, DenseNet style.  Hopefully both pilot
# and data symbol information will be used in EQ process.  
#
# Give it a few layers to approximate non-linear functions like cos,sin and 
# arg[] that are used in classical DSP ML. Network learns complex maths
# operations.
#
# Network also performs deframing, extracting just the data symbols
    
class EQ(nn.Module):
    def __init__(self, framer, EbNodB):
        super().__init__()

        self.framer = framer
        self.n_pilot = framer.n_pilot
        self.n_data = framer.n_data
        self.n_total = self.n_pilot + self.n_data
        self.EbNodB = EbNodB

        self.dense1 = nn.Linear(2*self.n_total, w1)
        self.dense2 = nn.Linear(w1+2*self.n_total, w1)
        self.dense3 = nn.Linear(w1+2*self.n_total, w1)
        self.dense4 = nn.Linear(w1+2*self.n_total, w1)
        self.dense5 = nn.Linear(w1+2*self.n_total, 2*self.n_data)

    # note rx_frame complex values passed in as real,imag pairs
    def equaliser(self, rx_frame):
        x = torch.relu(self.dense1(torch.cat([rx_frame],-1)))
        x = torch.relu(self.dense2(torch.cat([x, rx_frame],-1)))
        x = torch.relu(self.dense3(torch.cat([x, rx_frame],-1)))
        x = torch.relu(self.dense4(torch.cat([x, rx_frame],-1)))
        equalised_data = self.dense5(torch.cat([x, rx_frame],-1))
        return equalised_data

    # tx_data is complex, return real values real,imag pairs to suit loss function
    def forward(self, tx_data):
        tx_frame = self.framer.frame(tx_data)
        rx_frame, rx_data = self.framer.channel(tx_frame, self.EbNodB)

        tx_data_float = tofloat(tx_data)
        rx_frame_float = tofloat(torch.reshape(rx_frame,(rx_frame.shape[0],self.n_total)))
        rx_data_float = tofloat(rx_data)

        # run equaliser
        if args.eq == "bypass":
            rx_data_eq = rx_data_float
        if args.eq == "ml":
            rx_data_eq = self.equaliser(rx_frame_float)
        if args.eq == "dsp":
            rx_data_eq = self.framer.dsp_equaliser(rx_frame)
 
        return tx_data_float,rx_data_float,rx_data_eq
        
# sym and sym hat in float format (real,imag pairs)
def loss_phase_mse(sym_hat, sym):
    sym = sym[:,0] + 1j*sym[:,1]
    sym_hat = sym_hat[:,0] + 1j*sym_hat[:,1]
    error = torch.angle(sym*torch.conj(sym_hat))
    loss = torch.sum(error**2)
    return loss

if args.framer == 1:
    aframer = frame1()
if args.framer == 2:
    aframer = frame2()

model = EQ(aframer, args.EbNodB).to(device)
nb_params = sum(p.numel() for p in model.parameters())
print(f" {nb_params} weights")

if args.train:
    if args.loss_phase:
        loss_fn = loss_phase_mse
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    dataset = aQPSKDataset(n_syms, model.n_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Train model
    for epoch in range(args.epochs):
        sum_loss = 0.0
        for batch,(tx_data) in enumerate(dataloader):
            tx_data = tx_data.to(device)
            tx_data_float,rx_data_float,rx_data_eq = model(tx_data)
            loss = loss_fn(rx_data_eq,tx_data_float)
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
    bits = torch.sign(torch.rand(n_syms*model.n_data, bps)-0.5)
    tx_data = (bits[:,::2] + 1j*bits[:,1::2])/np.sqrt(2.0)
    tx_data = torch.reshape(tx_data,(n_syms,model.n_data))
    model.EbNodB = EbNodB
    with torch.no_grad():
        tx_data = tx_data.to(device)
        tx_data_float,rx_data_float,rx_data_eq = model(tx_data)
    tx_data_float = tx_data_float.cpu().numpy()
    rx_data_float = rx_data_float.cpu().numpy()
    rx_data_eq = rx_data_eq.cpu().numpy()

    n_errors = np.sum(-tx_data_float.flatten()*rx_data_eq.flatten()>0)
    n_bits = n_syms*model.n_data*bps
    BER = n_errors/n_bits
    print(f"EbNodB: {EbNodB:5.2f} n_bits: {n_bits:d} n_errors: {n_errors:d} BER: {BER:5.3f}")

    return BER, rx_data_float, rx_data_eq

if len(args.curve):
    EbNodB = np.array([0,1,2,3,4,5,6,7,8], dtype=np.float32)
    n_tests = len(EbNodB)
    curve = np.zeros((n_tests,2))
    for i in np.arange(0,n_tests):
        curve[i,0] = EbNodB[i]
        curve[i,1],rx_data_float,rx_data_eq = single_point(EbNodB[i], args.n_syms)
    np.savetxt(args.curve, curve)
else:
    # single point with scatter plot
    ber,rx_data_float,rx_data_eq = single_point(args.EbNodB, args.n_syms)
    if args.plots:
        plt.plot(rx_data_float[:,0],rx_data_float[:,1],'+')
        plt.plot(rx_data_eq[:,0],rx_data_eq[:,1],'+')
        plt.axis([-2,2,-2,2])
        plt.show()

