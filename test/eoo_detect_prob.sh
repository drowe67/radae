#!/bin/bash -e
#
# Measure probability of correct EOO detection for RADE V2.
# Steps through all.wav in 2-second segments, running one EOO trial per segment.
# For MPP, fading_adv is stepped across trials to sample different fade positions.
# Uses tx2.py -> ch -> rx2.py, so data and EOO both genuinely see the channel
# (fading, noise, SSB filtering), unlike inference.py's --end_of_over_v2 splice.
#
# Usage:
#   ./test/eoo_detect_prob.sh [--No <dBHz>] [--channel <awgn|mpp>] [--N <trials>]

No=-100
channel=awgn
N=20
seg_len=9   # seconds per trial, make it long compared to fading duration
            # and such that wav_dur/seg_len and g_mpp_dur/seg_len are not integers -
            # so sucessive trial are not aligned with either file
verbose=0

function print_help {
    echo
    echo "Measure RADE V2 EOO detection probability"
    echo
    echo "  usage: ./test/eoo_detect_prob.sh [--No dBHz] [--channel awgn|mpp] [--N trials] [--verbose]"
    echo "  example: ./test/eoo_detect_prob.sh --No -25 --channel mpp --N 20"
    echo
    exit
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --No)      No=$2;      shift 2 ;;
        --channel) channel=$2; shift 2 ;;
        --N)       N=$2;       shift 2 ;;
        --verbose) verbose=1;  shift ;;
        -h|--help) print_help ;;
        *) echo "Unknown argument: $1"; print_help ;;
    esac
done

if [ "$channel" = "mpp" ]; then
    chan_args="--mpp --fading_dir ."
    g_step=$seg_len
    test/make_g.sh
elif [ "$channel" = "awgn" ]; then
    chan_args=""
else
    echo "Unknown channel: $channel (use awgn or mpp)"
    exit 1
fi

wav_dur=$(soxi -D wav/all.wav)
# 2 complex numbers per time step
g_mpp_dur=$(python3 -c "import os; print(os.path.getsize('g_mpp.f32')//(2*2*4*8000))")

detected=0
snr3k_sum=0
tx_tmp=$(mktemp /tmp/eoo_detect_tx_XXXXXX.f32)
rx_tmp=$(mktemp /tmp/eoo_detect_XXXXXX.f32)
wav_tmp=$(mktemp /tmp/eoo_seg_XXXXXX.wav)
ch_log=$(mktemp /tmp/eoo_detect_ch_XXXXXX.log)
trap "rm -f $tx_tmp $rx_tmp $wav_tmp $ch_log" EXIT

echo "EOO detection probability: No=$No channel=$channel N=$N seg_len=${seg_len}s"

for i in $(seq 1 $N); do
    offset=$(python3 -c "print((($i-1)*$seg_len) % (int($wav_dur)-$seg_len))")
    sox wav/all.wav $wav_tmp trim $offset $seg_len

    fading_adv_args=""
    g_off=0
    if [ "$channel" = "mpp" ]; then
        g_off=$(python3 -c "print((($i-1)*$g_step) % ($g_mpp_dur-$seg_len))")
        fading_adv_args="--fading_adv $g_off"
    fi

    ./tx2.sh 250725/checkpoints/checkpoint_epoch_200.pth $wav_tmp ${tx_tmp} 2>/dev/null
    cat ${tx_tmp} | python3 f32toint16.py --real --scale 8192 | \
        build/src/ch - - --No $No $chan_args $fading_adv_args 2>$ch_log | \
        python3 int16tof32.py --zeropad > $rx_tmp
    snr3k=$(grep -oP 'SNR3k\(dB\):\s*\K[-0-9.]+' $ch_log)
    snr3k_sum=$(python3 -c "print($snr3k_sum + $snr3k)")

    if [ "$verbose" -eq 1 ]; then
        result=$(./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync $rx_tmp /dev/null \
            --latent-dim 56 --w1_dec 128 --correct_time_offset -8 2>&1 | tee /dev/stderr | grep -c "EOO detected" || true)
    else
        result=$(./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync $rx_tmp /dev/null \
            --latent-dim 56 --w1_dec 128 --correct_time_offset -8 2>&1 | grep -c "EOO detected" || true)
    fi

    if [ "$result" -gt 0 ]; then
        detected=$((detected + 1))
        echo "  trial $i: DETECTED (SNR3k=${snr3k}dB offset=${offset}s fading_adv=$g_off)"
    else
        echo "  trial $i: missed   (SNR3k=${snr3k}dB offset=${offset}s fading_adv=$g_off)"
    fi
done

prob=$(python3 -c "print(f'{$detected/$N:.2f}')")
snr3k_mean=$(python3 -c "print(f'{$snr3k_sum/$N:.2f}')")
echo "Result: $detected/$N detected, P(detect) = $prob, mean SNR3k = ${snr3k_mean}dB"
