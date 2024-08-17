#!/bin/bash -x
#
# test measurement of C/No using chirp on MPP channel
#
# test usage:
#
#   ~/radae main $ ./test/chirp_mpp.sh ~/codec2-dev/build_linux/ -16

if [ $# -ne 2 ]; then
  echo "usage: $0 /path/to/codec2-dev/build No"
  exit 1
fi

CODEC2_DEV_BUILD_DIR=$1
No=$2
chirp_duration=4
silence_duration=3

which ${CODEC2_DEV_BUILD_DIR}/src/ch >/dev/null || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

source test/make_g.sh
cp -f g_mpp.f32 fast_fading_samples.float
chirp_f32=$(mktemp)
chirp_int16=$(mktemp)
silence_int16=$(mktemp)
chirp_pad_int16=$(mktemp)
chirp_noise_int16=$(mktemp)
chirp_noise_f32=$(mktemp)
ch_log=$(mktemp)
python3 chirp.py ${chirp_f32} ${chirp_duration}
cat ${chirp_f32} | python3 f32toint16.py --real > ${chirp_int16}
# pad either side with a few seconds of seconds of silence
dd if=/dev/zero of=/dev/stdout bs=16000 count=${silence_duration} > ${silence_int16}
cat ${silence_int16} ${chirp_int16} ${silence_int16} > ${chirp_pad_int16}
# Note --ssbfilt 1 (default) removes -ve freq part of complex noise 
# which ensures C/No remains unaffected by real() operation at output 
${CODEC2_DEV_BUILD_DIR}/src/ch ${chirp_pad_int16} ${chirp_noise_int16} --No ${No} --mpp --fading_dir . --after_fade 2>${ch_log}
cat ${chirp_noise_int16} | python3 int16tof32.py  --zeropad > ${chirp_noise_f32}
est_log=$(mktemp)
python3 est_CNo.py ${chirp_noise_f32} >${est_log}

# check two estimates are about the same

CNodB_ch=$(cat ${ch_log} | grep "C/No" | tr -s ' ' | cut -d' ' -f5)
CNodB_est=$(cat ${est_log} | grep "Measured:" | tr -s ' ' | cut -d' ' -f3)
python3 <<EOF
import sys
import numpy as np
# ch C est will read low, as C is averaged across entire sample including silence
CNodB_ch = ${CNodB_ch} + 10*np.log10((2*${silence_duration}+${chirp_duration})/${chirp_duration})
print(f"{CNodB_ch:5.2f}")
if abs(CNodB_ch-${CNodB_est}) > 1.0:
  sys.exit(1)
EOF
if [ $? -eq 1 ]; then
  exit 1
fi

# check time estimate is OK

chirp_start=$(cat ${est_log} | grep "Measured:" | tr -s ' ' | cut -d' ' -f2)
python3 <<EOF
import sys
if abs(${chirp_start} - ${silence_duration}) > 0.5:
  sys.exit(1)
EOF
if [ $? -eq 1 ]; then
  exit 1
fi

