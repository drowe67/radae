#!/bin/bash
#
# RADE V2 wrong carrier detector: compare the rade_c receiver (radae_rx --v2) with the
# Python reference (rx2.py) on the same received signal. PASS if the per-symbol rho values
# match closely and the acquisition veto and mistune decisions are identical.
#
# rx2.py runs with its AGC off by default (rade_c on), so the C receiver is run with --agc 0
# to compare like with like.
#
# usage: ./test/v2_c_wrong_k_compare.sh rade_c_build_src freq_offset_Hz [inference.py channel options]
#   e.g. ./test/v2_c_wrong_k_compare.sh ~/rade_c/build/src -45 --g_file g_mpp.f32 --EbNodB 6

OPUS=build/src
PATH=${PATH}:${OPUS}
RADE_C=$1
f_off=$2
shift; shift

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
--peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx rx_wrong_k_cmp.f32 \
--prepend_noise 1 --append_noise 2 --freq_offset $f_off "$@"

python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync rx_wrong_k_cmp.f32 /dev/null \
--latent-dim 56 --w1_dec 128 --quiet --write_wrong_k wrong_k_py.f32 2> rx_wrong_k_cmp_py.log
cat rx_wrong_k_cmp.f32 | ${RADE_C}/radae_rx --v2 -v 0 --agc 0 --write_wrong_k wrong_k_c.f32 > /dev/null 2> rx_wrong_k_cmp_c.log
grep "n_acq\|n_veto" rx_wrong_k_cmp_py.log rx_wrong_k_cmp_c.log

python3 -c "
import numpy as np
# rx2.py logs symbol s at index s (counter incremented before processing), radae_rx at s-1
p = np.fromfile('wrong_k_py.f32', np.float32).reshape(-1,3)[1:]
c = np.fromfile('wrong_k_c.f32', np.float32).reshape(-1,3)
n = min(len(p), len(c)); p, c = p[:n], c[:n]
d = np.abs(p[:,0] - c[:,0])
veto_err = np.count_nonzero(p[:,1] != c[:,1]); mistune_err = np.count_nonzero(p[:,2] != c[:,2])
print(f'symbols: {n} rho max diff: {d.max():.2e} vetoes py: {int(p[:,1].sum())} c: {int(c[:,1].sum())} '
      f'veto mismatches: {veto_err} mistune mismatches: {mistune_err}')
ok = d.max() < 0.01 and veto_err == 0 and mistune_err == 0 and p[:,1].sum() > 0
print('PASS' if ok else 'FAIL')
"
