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

source test/make_g_mpp.sh
cp -f g_mpp.f32 fast_fading_samples.float
chirp_f32=$(mktemp)
chirp_int16=$(mktemp)
chirp_noise_int16=$(mktemp)
chirp_noise_f32=$(mktemp)
python3 chirp.py ${chirp_f32} 4
python3 f32toint16.py ${chirp_f32} ${chirp_int16} --real
ch_log=$(mktemp)
# Note --ssbfilt 1 (default) removes -ve freq part of complex noise 
# which ensures C/No remains unaffected by real() operation at output 
${CODEC2_DEV_BUILD_DIR}/src/ch ${chirp_int16} ${chirp_noise_int16} --No ${No} --mpp --fading_dir . --after_fade 2>${ch_log}
python3 int16tof32.py ${chirp_noise_int16} ${chirp_noise_f32} --zeropad
est_log=$(mktemp)
python3 est_CNo.py ${chirp_noise_f32} >${est_log}

CNodB_ch=$(cat ${ch_log} | grep "C/No" | tr -s ' ' | cut -d' ' -f5)
CNodB_est=$(cat ${est_log} | grep "Measured:" | tr -s ' ' | cut -d' ' -f2)
python3 -c "if abs(${CNodB_ch}-${CNodB_est}) < 1.0: print('PASS')"
