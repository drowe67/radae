import sys
import numpy as np

# bits are in float form, e.g. +/-1 or +/-1000
tx_bits = np.fromfile(sys.argv[1], dtype=np.float32)
rx_bits = np.fromfile(sys.argv[2], dtype=np.float32)
bits_per_frame = len(tx_bits)
n_frames = len(rx_bits)//bits_per_frame
n_ok_frames = 0
for f in range(n_frames):
    n_errors = sum(rx_bits[f*bits_per_frame:(f+1)*bits_per_frame]*tx_bits < 0)
    ber = n_errors/bits_per_frame
    print(f"frame received! BER: {ber:5.2f}")
    if ber < 0.05:
        n_ok_frames += 1
print(f"EOO frames  received: {n_frames} n_ok_frames: {n_ok_frames}", file=sys.stderr)
if n_ok_frames:
    print("PASS", file=sys.stderr)
