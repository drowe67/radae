#!/bin/bash -x
#
# RADE V2 spot test for "4 point tests" - use extra arguments to set channel type

OPUS=build/src
PATH=${PATH}:${OPUS}

features_out=features_out_rx2.f32
# larger more realistic for low SNR test
delta=${delta:-0.01}

function print_help {
    echo
    echo "RADE V2 spot test helper script"
    echo
    echo "  usage ./test/v2_spot.sh [--g_file file_name] [--EbNodB value] [--prepend_noise Nsecs] [rx2.sh options]"
    echo "  for example:"
    echo "       ./test/v2_spot.sh "
    echo "       ./test/v2_spot.sh --g_file g_mpp.f32 --EbNodB 6 "
    echo "       ./test/v2_spot.sh "
    echo "       ./test/v2_spot.sh --EbNodB 1 --gain 0.1 --agc"
    echo
    exit
}

# strip of args for inference.sh
g_file=""
a_g_file=""
EbNodB=""
a_EbNodB_value=""
a_prepend_noise=1
sine_amp=""
a_sine_amp=""
sine_freq=""
a_sine_freq=""
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
    --EbNodB)
        EbNodB="--EbNodB"
        a_EbNodB_value="$2"	
        shift
        shift
    ;;
    --prepend_noise)
        a_prepend_noise="$2"	
        shift
        shift
    ;;
    --sine_amp)
        sine_amp="--sine_amp"
        a_sine_amp="$2"	
        shift
        shift
    ;;
    --sine_freq)
        sine_freq="--sine_freq"
        a_sine_freq="$2"	
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

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null --rate_Fs --latent-dim 56 \
--peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx 250725_rx.f32 \
--prepend_noise $a_prepend_noise --append_noise 2 --freq_offset 25 --correct_freq_offset \
$g_file $a_g_file $EbNodB $a_EbNodB_value $sine_amp $a_sine_amp $sine_freq $a_sine_freq

# debug tips:
# 1) Use Octave to plot internal states, e.g. 
# octave> figure(1); clf; sig_det=load_raw('sig_det.int16'); plot(sig_det); state=load_raw('state.int16'); \
# hold on; plot(state*1.5); hold off; \
# figure(2); freq_offset_smooth=load_f32('freq_offset_smooth.f32',1); plot(freq_offset_smooth);
# 2) remove --quiet and look for state transitions (e.g. back to noise), which upsets alignment for loss.py
./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync 250725_rx.f32 /dev/null --latent-dim 56 \
--w1_dec 128 --hangover 100 --correct_time_offset -8 --quiet \
--write_sig_det sig_det.int16 --write_state state.int16 --write_freq_offset_smooth freq_offset_smooth.f32 \
--write_frame_sync frame_sync.f32 $@

# debug tip: run from cmd line with --plot
python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_out_rx2.f32 \
--clip_start 100 --clip_end 300 --compare --delta $delta