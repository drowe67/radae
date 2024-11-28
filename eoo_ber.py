import sys
import numpy as np

# bits are in float form, e.g. +/-1 or +/-1000
tx_bits = np.fromfile(sys.argv[1], dtype=np.float32)
rx_bits = np.fromfile(sys.argv[2], dtype=np.float32)
n_bits = len(tx_bits)
n_errors = sum(rx_bits*tx_bits < 0)
ber = n_errors/n_bits
print(f"EOO data n_bits: {n_bits} n_errors: {n_errors} BER: {ber:5.2f}", file=sys.stderr)
if ber < 0.05:
    print("PASS", file=sys.stderr)
