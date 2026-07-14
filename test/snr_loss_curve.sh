#!/bin/bash
#
# RADE V2 loss vs SNR curve using all.wav over AWGN channel.
# Runs both Python rx (rx2.sh) and C port rx (radae_nopy) for comparison.
# Useful as a software-only reference for calibrating OTC integration results.
# Generates a PNG plot of loss vs SNR3k.
#
# Usage: ./test/snr_loss_curve.sh [path/to/radae_nopy/build]
# Run from the radae repo root directory.

set -e

MODEL=250725
CHECKPOINT=${MODEL}/checkpoints/checkpoint_epoch_200.pth
SYNC=${MODEL}a_ml_sync
WAV=wav/all.wav
RESULTS_PY=${MODEL}_awgn_loss_SNR3k_py.txt
RESULTS_C=${MODEL}_awgn_loss_SNR3k_c.txt
PNG=${MODEL}_awgn_loss_SNR3k.png
NOPY_BUILD=${1:-~/radae_nopy/build}
RADAE_RX=${NOPY_BUILD}/src/radae_rx

# EbNodB range: calibration tool, focus on usable SNR range
EbNodB_list="6 9 12 15 18 21 24 27 100"

rm -f ${RESULTS_PY} ${RESULTS_C}
echo "Model: ${MODEL}  WAV: ${WAV}  Channel: AWGN"
printf "%-10s %-12s %-12s\n" "SNR3k" "loss_py" "loss_c"

for EbNodB in ${EbNodB_list}; do
    log=$(./inference.sh ${CHECKPOINT} ${WAV} /dev/null \
        --rate_Fs --latent-dim 56 --peak --cp 0.004 \
        --time_offset -16 --correct_time_offset -16 \
        --auxdata --w1_dec 128 --write_rx 250725_rx.f32 \
        --prepend_noise 1 --append_noise 2 \
        --freq_offset 25 --correct_freq_offset \
        --EbNodB ${EbNodB} 2>&1)

    SNR3k=$(echo "${log}" | grep "Measured:" | tr -s ' ' | cut -d' ' -f4)

    # Python rx
    ./rx2.sh ${CHECKPOINT} ${SYNC} 250725_rx.f32 /dev/null --quiet
    loss_py=$(python3 loss.py features_in.f32 features_out_rx2.f32 \
        --clip_start 100 --clip_end 300 2>/dev/null | grep "loss:" | awk '{print $2}')

    # C port rx
    cat 250725_rx.f32 | ${RADAE_RX} --v2 -v 0 > features_rx_c.f32 2>/dev/null
    loss_c=$(python3 loss.py features_in.f32 features_rx_c.f32 \
        --clip_start 100 --clip_end 300 2>/dev/null | grep "loss:" | awk '{print $2}')

    printf "%-10s %-12s %-12s\n" "${SNR3k}" "${loss_py}" "${loss_c}"
    printf "%s\t%s\n" "${SNR3k}" "${loss_py}" >> ${RESULTS_PY}
    printf "%s\t%s\n" "${SNR3k}" "${loss_c}"  >> ${RESULTS_C}
done

python3 - <<EOF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

py  = np.loadtxt('${RESULTS_PY}')
c   = np.loadtxt('${RESULTS_C}')

plt.figure(figsize=(8,5))
plt.plot(py[:,0], py[:,1], 'b-o', label='RADE V2 Python rx (rx2.sh)')
plt.plot(c[:,0],  c[:,1],  'r-s', label='RADE V2 C port rx (radae_rx)')
plt.xlabel('SNR3k (dB)')
plt.ylabel('Loss')
plt.title('RADE V2 Loss vs SNR3k (AWGN)')
plt.xlim(-2, 22)
plt.yscale('log')
ax = plt.gca()
ax.set_yticks([0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('${PNG}')
print('Generated: ${PNG}')
EOF
