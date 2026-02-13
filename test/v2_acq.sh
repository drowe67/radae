#!/bin/bash -x
#
# RADE V2 helper script to perform acquisition tests, e.g. false acquisition on noise, or noise plus sine wave.

OPUS=build/src
PATH=${PATH}:${OPUS}

function print_help {
    echo
    echo "RADE V2 acquisition noise test helper script"
    echo
    echo "  usage ./test/v2_acq.sh --codec2_dev ~/codec2_dev/build [--nsecs Nsecs] [--sine] [rx2.sh options]"
    echo
    exit
}

CODEC2_DEV_BUILD_DIR="${HOME}/codec2-dev/build_linux"

n_secs=60

sine=0
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --n_secs)
        n_secs="$2"	
        shift
        shift
    ;;
    --codec2_dev)
        $CODEC2_DEV_BUILD_DIR="$2"	
        shift
        shift
    ;;
    --sine)
        sine=1
        shift
    ;;
    -h)
        print_help	
    ;;
    *)
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

ch_in=$(mktemp)
if [ $sine -eq 1 ]; then
  ${CODEC2_DEV_BUILD_DIR}/misc/mksine $ch_in 1000 ${n_secs}
else
  dd if=/dev/zero of=/dev/stdout bs=16000 count=${n_secs} > $ch_in
fi
cat $ch_in | ${CODEC2_DEV_BUILD_DIR}/src/ch - - --No -20 | python3 int16tof32.py --zeropad > rx.f32

./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync rx.f32 /dev/null --latent-dim 56 \
--w1_dec 128 --hangover 100 --correct_time_offset -8 --quiet --write_Ry_smooth Ry_smooth.f32 --write_Ry_max Ry_max.f32 --write_state state.int16 $@
