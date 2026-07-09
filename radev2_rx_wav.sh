#!/bin/bash -x
#
# radev2_rx_wav.sh - Decode a RADE V2 OTA WAV recording and generate diagnostic plots
#
# Usage:
#   ./radev2_rx_wav.sh <rx.wav> [extra rx2.py args]
#
# Example:
#   ./radev2_rx_wav.sh ~/Downloads/kiwi_sdr_rx.wav
#
# Output (all artefacts stored alongside rx.wav in <basename>/ directory):
#   <basename>_rade2.wav  - decoded speech
#   <basename>_plots.png  - sync state, timing, freq offset, gain, SNR plots
#   state.int16           - sync state machine output
#   delta_hat.f32         - timing estimate
#   delta_hat_g.f32       - timing estimate (global)
#   freq_offset.f32       - frequency offset estimate
#   gain.f32              - AGC gain
#   snr_est.f32           - SNR estimate (dB)
#   report.txt            - rx2.py stderr log

set -e

OPUS=build/src
PATH=${PATH}:${OPUS}

MODEL=250725/checkpoints/checkpoint_epoch_200.pth
MODEL_SYNC=250725a_ml_sync

if [ $# -lt 1 ]; then
    echo "usage: ./radev2_rx_wav.sh <rx.wav> [extra rx2.py args]"
    exit 1
fi

RXWAV="$1"; shift
OUTDIR="$(dirname "$RXWAV")/$(basename "${RXWAV%.*}")"
BASENAME="${OUTDIR}/$(basename "${RXWAV%.*}")"
mkdir -p "$OUTDIR"

# Resample to 8kHz and convert to f32 IQ (zeropad Q=0)
TMP_F32=$(mktemp /tmp/radev2_rx_XXXXXX.f32)
sox "$RXWAV" -t .s16 -r 8000 -c 1 - | python3 int16tof32.py --zeropad > "$TMP_F32"

# Decode
VERBOSE=0
for arg in "$@"; do [ "$arg" = "--verbose" ] && VERBOSE=1; done

RX2_ARGS=(--gain 1.22E-4 --agc
    --write_state       "$OUTDIR/state.int16"
    --write_delta_hat   "$OUTDIR/delta_hat.f32"
    --write_delta_hat_g "$OUTDIR/delta_hat_g.f32"
    --write_freq_offset "$OUTDIR/freq_offset.f32"
    --write_gain        "$OUTDIR/gain.f32"
    --write_snr_est     "$OUTDIR/snr_est.f32"
    "$@")

if [ $VERBOSE -eq 1 ]; then
    ./rx2.sh "$MODEL" "$MODEL_SYNC" "$TMP_F32" "${BASENAME}_rade2.wav" \
        "${RX2_ARGS[@]}" 2>"$OUTDIR/report.txt"
else
    ./rx2.sh "$MODEL" "$MODEL_SYNC" "$TMP_F32" "${BASENAME}_rade2.wav" \
        "${RX2_ARGS[@]}" 2>&1 | grep -v '^+' | \
        sed 's/ nin: [0-9]*//; s/ sine: [0-9]*//; s/ c: *[0-9]*//; s/ nsd: [0-9]*//; s/ nsf: [0-9]*//; s/ c1: *[0-9]*//; s/ fs: [0-9]*//; s/ delta_hat: *[0-9]*//; s/ delta_hat_g: *[0-9]*//; s/ f_off_g: *[-0-9.]*//; s/ Ry_max: *[-0-9.]*//; s/ Ry_min: *[-0-9.]*//; s/ corr: *[-0-9.]*//' \
        > "$OUTDIR/report.txt"
fi

rm -f "$TMP_F32"

# Generate diagnostic plots
octave-cli -qf --path "${PWD}" --eval \
    "radae_plots; plot_v2_logs('${BASENAME}_plots.png', \
     '$OUTDIR/state.int16', '$OUTDIR/delta_hat.f32', '$OUTDIR/delta_hat_g.f32', \
     '$OUTDIR/freq_offset.f32', '$OUTDIR/gain.f32', '$OUTDIR/snr_est.f32'); quit"

echo "Output in: $OUTDIR"
