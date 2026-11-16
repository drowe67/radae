#!/bin/bash -x
#
# RADE V2 C receiver: non-finite input (NaN, Inf, huge values) must not stop the receiver.
# All C signal chain: radae_tx --v2 -> real int16 -> ch -> radae_rx --v2, two overs with gaps.
# Bad samples are injected in the first over, just after it, and in the second over.
# PASS if the damaged file gives the same number of acquisitions and valid outputs as the
# clean file, all output features are finite, and each bad block was counted.
#
# usage: ./test/v2_c_nonfinite.sh rade_c_build_src

RADE_C=$1

build/src/lpcnet_demo -features wav/brian_g8sez.wav features_nf.f32
cat features_nf.f32 | ${RADE_C}/radae_tx --v2 > tx_nf1.f32
python3 -c "
import numpy as np
x = np.fromfile('tx_nf1.f32', dtype=np.float32); z = np.zeros(2*8000, dtype=np.float32)
np.concatenate([z, x, z, x, z]).tofile('tx_nf.f32')"
cat tx_nf.f32 | python3 f32toint16.py --real --scale 8192 > tx_nf.raw
build/src/ch tx_nf.raw rx_nf.raw --No -30
python3 int16tof32.py --zeropad < rx_nf.raw > rx_nf_clean.f32

# one NaN at 4 s (in the first over), one Inf at 12.5 s (gap), one 1E30 at 17 s (second over)
python3 -c "
import numpy as np
x = np.fromfile('rx_nf_clean.f32', dtype=np.float32)
for t, v in [(4.0, np.nan), (12.5, np.inf), (17.0, 1E30)]:
    x[2*int(t*8000)] = v
x.tofile('rx_nf_bad.f32')"

cat rx_nf_clean.f32 | ${RADE_C}/radae_rx --v2 -v 0 > features_nf_clean.f32 2> rx_nf_clean.log
cat rx_nf_bad.f32   | ${RADE_C}/radae_rx --v2 -v 0 > features_nf_bad.f32   2> rx_nf_bad.log
grep "valid outputs\|n_acq\|n_nonfinite" rx_nf_clean.log rx_nf_bad.log

python3 -c "
import numpy as np, re
def stats(fn):
    s = open(fn).read()
    return (int(re.search(r'(\d+) valid outputs', s).group(1)), int(re.search(r'n_acq: (\d+)', s).group(1)),
            int(re.search(r'n_nonfinite: (\d+)', s).group(1)))
c = stats('rx_nf_clean.log'); b = stats('rx_nf_bad.log')
f = np.fromfile('features_nf_bad.f32', dtype=np.float32)
ok = c[1] == 2 and b[0] == c[0] and b[1] == c[1] and b[2] == 3 and c[2] == 0 and np.all(np.isfinite(f))
print(f'clean: valid {c[0]} n_acq {c[1]} n_nonfinite {c[2]}  bad: valid {b[0]} n_acq {b[1]} n_nonfinite {b[2]}  finite {np.all(np.isfinite(f))}')
exit(0 if ok else 1)"
if [ $? -eq 0 ]; then echo "PASS"; else echo "FAIL"; fi
