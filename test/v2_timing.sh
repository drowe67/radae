#!/bin/bash -x
#
# RADE V2 Jan 2026 timing investigation script 

OPUS=build/src
PATH=${PATH}:${OPUS}
delta=${delta:-0.01}

function print_help {
    echo
    echo "RADE V2 Jan 2026 timing investigation helper script"
    echo
    echo "  usage ./test/v2_timing.sh [--g_file file_name] [rx.sh options]"
    echo "  for example:"
    echo "       ./test/v2_timing.sh --correct_time_offset -22"
    echo "       ./test/v2_timing.sh --g_file mpp_low.f32 --correct_time_offset -8 --fix_delta_hat 32"
    echo
    exit
}

# strip of args for inference.sh
g_file=""
a_g_file=""
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --g_file)
        g_file="--g_file"
        a_g_file="$2"	
        shift
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

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
--peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx 250725_rx.f32 \
$g_file $a_g_file

./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync 250725_rx.f32 /dev/null --latent-dim 56 \
--w1_dec 128 --no_bpf --write_delta_hat delta_hat.int16 --write_delta_hat_pp delta_hat_pp.int16 $@

python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_out_rx2.f32 --clip_start 25 --compare --delta $delta
