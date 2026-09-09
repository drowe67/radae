#!/bin/bash -e
#
# Measure false EOO detection rate for RADE V2.
# Runs N independent trials using the full wav with tx2.py --no_eoo (no real
# EOO frame present), counts how many times rx2.py falsely triggers
# "EOO detected" per trial. Each trial has a fresh noise realisation; for
# MPP, fading_adv steps through the channel file to sample different fade
# positions. Uses tx2.py -> ch -> rx2.py, same channel path as
# eoo_detect_prob.sh, for consistency.
#
# Usage:
#   ./test/eoo_false_prob.sh [--No <dBHz>] [--channel <awgn|mpp>] [--N <trials>]

No=-100
channel=awgn
N=20
wav=wav/all.wav

function print_help {
    echo
    echo "Measure RADE V2 EOO false detection rate"
    echo
    echo "  usage: ./test/eoo_false_prob.sh [--No dBHz] [--channel awgn|mpp] [--N trials] [--wav wavefile]"
    echo "  example: ./test/eoo_false_prob.sh --No -25 --channel mpp --N 20"
    echo
    exit
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --No)      No=$2;      shift 2 ;;
        --channel) channel=$2; shift 2 ;;
        --N)       N=$2;       shift 2 ;;
        --wav)     wav=$2;     shift 2 ;;
        -h|--help) print_help ;;
        *) echo "Unknown argument: $1"; print_help ;;
    esac
done

wav_dur=$(soxi -D $wav)

if [ "$channel" = "mpp" ]; then
    chan_args="--mpp --fading_dir ."
    g_step=$(python3 -c "print(int($wav_dur))")   # step by one wav length per trial
    test/make_g.sh
    g_mpp_dur=$(python3 -c "import os; print(os.path.getsize('g_mpp.f32')//(2*2*4*8000))")
elif [ "$channel" = "awgn" ]; then
    chan_args=""
else
    echo "Unknown channel: $channel (use awgn or mpp)"
    exit 1
fi

total_false=0
snr3k_sum=0
tx_tmp=$(mktemp /tmp/eoo_false_tx_XXXXXX.f32)
rx_tmp=$(mktemp /tmp/eoo_false_XXXXXX.f32)
ch_log=$(mktemp /tmp/eoo_false_ch_XXXXXX.log)
trap "rm -f $tx_tmp $rx_tmp $ch_log" EXIT

echo "EOO false detection rate: No=$No channel=$channel N=$N"

for i in $(seq 1 $N); do
    fading_adv_args=""
    g_off=0
    if [ "$channel" = "mpp" ]; then
        g_off=$(python3 -c "print((($i-1)*$g_step) % ($g_mpp_dur - $wav_dur))")
        fading_adv_args="--fading_adv $g_off"
    fi

    ./tx2.sh 250725/checkpoints/checkpoint_epoch_200.pth $wav ${tx_tmp} --no_eoo 2>/dev/null
    cat ${tx_tmp} | python3 f32toint16.py --real --scale 8192 | \
        build/src/ch - - --No $No $chan_args $fading_adv_args 2>$ch_log | \
        python3 int16tof32.py --zeropad > $rx_tmp
    snr3k=$(grep -oP 'SNR3k\(dB\):\s*\K[-0-9.]+' $ch_log)
    snr3k_sum=$(python3 -c "print($snr3k_sum + $snr3k)")

    count=$(./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync $rx_tmp /dev/null \
        --latent-dim 56 --w1_dec 128 --correct_time_offset -8 2>&1 | grep -c "EOO detected" || true)

    total_false=$((total_false + count))
    echo "  trial $i: $count false triggers (SNR3k=${snr3k}dB fading_adv=$g_off)"
done

snr3k_mean=$(python3 -c "print(f'{$snr3k_sum/$N:.2f}')")
echo "Mean SNR3k = ${snr3k_mean}dB"
if [ "$total_false" -gt 0 ]; then
    avg_time=$(python3 -c "print(f'{$N*$wav_dur/$total_false:.1f}')")
    echo "Result: $total_false false triggers over $N trials, avg time between false triggers = ${avg_time} s"
else
    echo "Result: 0 false triggers over $N trials"
fi
