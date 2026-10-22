#!/bin/bash
#
# RADE V2 C receiver sample clock offset test (rade_c Issue #27)
#
# inference.py Tx -> optional channel -> resample to rx_Fs (Tx/Rx sample clock
# offset) -> C radae_rx. Checks loss, and optionally the rade_clock_offset_ppm()
# estimate printed by radae_rx.
#
# Usage: ./test/v2_c_samclk.sh rade_c_build_src rx_Fs loss_thresh [expected_ppm ppm_tol]
#   e.g. +125 ppm, high SNR, check loss < 0.10 and ppm estimate within 15 of 125:
#        ./test/v2_c_samclk.sh ~/rade_c/build/src 8001 0.10 125 15
#
# INF_OPTS="..."  extra inference.py options (e.g. channel), RX_OPTS="..." extra
# radae_rx options (e.g. --timing_iir)

RADE_C=$1
rx_Fs=$2
loss_thresh=$3
expected_ppm=$4
ppm_tol=$5
INF_OPTS=${INF_OPTS:-}
RX_OPTS=${RX_OPTS:-}

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
  --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
  --auxdata --w1_dec 128 --write_rx rx_v2_samclk.f32 \
  --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset $INF_OPTS

# rx sample clock offset (same method as test/v2_spot.sh)
cat rx_v2_samclk.f32 | python3 f32toint16.py --scale 8192 | \
  sox -t .s16 -r 8000 -c 2 - -t .s16 -r $rx_Fs -c 2 - | \
  python3 int16tof32.py --scale 1.22E-4 > rx_v2_samclk_fs.f32

cat rx_v2_samclk_fs.f32 | ${RADE_C}/radae_rx --v2 -v 0 $RX_OPTS > features_rx_v2_c.f32 2> radae_rx_samclk.log
cat radae_rx_samclk.log

python3 loss.py features_in.f32 features_rx_v2_c.f32 --loss_test $loss_thresh --clip_start 100 --clip_end 300 || exit 1

if [ -n "$expected_ppm" ]; then
  ppm=$(grep "clock_offset_ppm:" radae_rx_samclk.log | cut -d' ' -f2)
  python3 -c "
ppm=float('$ppm'); exp=float('$expected_ppm'); tol=float('$ppm_tol')
print(f'clock offset estimate: {ppm:.1f} ppm expected: {exp:.1f} +/- {tol:.1f}')
print('PASS' if abs(ppm-exp) <= tol else 'FAIL')"
fi
