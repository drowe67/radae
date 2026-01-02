#!/bin/bash -x
#
# Test ota_test.sh and check calibration of chirp based C/No estimation. Note C/No = P/No for chirp,
# as PAPR is zero.
#
# test usage:
#
#   ~/radae main $ ./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -20

if [ $# -lt 2 ]; then
  echo "usage: $0 /path/to/codec2-dev/build No [ch options]"
  exit 1
fi

CODEC2_DEV_BUILD_DIR=$1
No=$2
loss_thresh=$3
shift; shift;
GAIN=0.25 # allow some headroom for noise and fading to prevent clipping
silence_duration=1

printf "\nMake fading samples .... \n\n"
source test/make_g.sh
cp -f g_mpp.f32 fast_fading_samples.float

printf "\nGenerate tx file and add noise ... \n\n"
./ota_test.sh -x wav/brian_g8sez.wav --peak
# add 1 second of silence to start to give est_CNo.py a work out
dd if=/dev/zero of=/dev/stdout bs=16000 count=${silence_duration} | sox -t .s16 -r 8000 -c 1 - sil.wav
sox sil.wav tx.wav tx_pad.wav
${CODEC2_DEV_BUILD_DIR}/src/ch tx_pad.wav - --gain ${GAIN} --No ${No} --after_fade --fading_dir . $@ | sox -t .s16 -r 8000 -c 1 - rx.wav

printf "\nRun chirp only through 'ch' to get reference estimate of C/No ... \n\n"
ch_log=$(mktemp)
sox tx.wav -t .s16 - trim 0 4 | \
~/codec2-dev/build_linux/src/ch - /dev/null --gain ${GAIN} --No ${No} --after_fade --fading_dir . $@ 2>${ch_log}
CNodB_ch=$(cat ${ch_log} | grep "C/No" | tr -s ' ' | cut -d' ' -f5)

printf "\nRun Rx and check ML "loss" is OK ... \n\n"
rm -f features_rx_out_rx1.f32
rx_log=$(mktemp)
./ota_test.sh -d -r rx.wav >${rx_log}
# --clip_end has side effect of increasing range of time_alignment, might be a better way to do that
python3 loss.py features_in.f32 features_out_rx1.f32 --loss_test ${loss_thresh} --clip_start 150 --clip_end 150 | tee /dev/stderr | grep "PASS" 
if [ $? -ne 0 ]; then
  exit 1
fi

printf "\nCheck C/No estimates close ...\n\n"
CNodB_est=$(cat ${rx_log} | grep "Measured:" | tr -s ' ' | cut -d' ' -f3)
if [ $? -ne 0 ]; then
  exit 1
fi
printf "\n\n"
cat ${rx_log} | grep "Measured:"