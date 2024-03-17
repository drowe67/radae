# utils.sh
#
# Bash include file with reusable utility functions
# Approximation of Hilbert clipper type compressor.  Could do with some HF boost

function analog_compressor {
    input_file=$1
    output_file=$2
    gain=$3
    cat $input_file | ch - - 2>/dev/null | \
    ch - - --No -100 --clip 16384 --gain $gain 2>/dev/null | \
    # final line prints peak and CPAPR for SSB
    ch - - --clip 16384 |
    sox -t .s16 -r 8000 -c 1 -v 0.85 - -t .s16 $output_file
}

function measure_rms() {
    ch_log=$(mktemp)
    ch $1 /dev/null 2>${ch_log}
    rms=$(cat $ch_log | grep "RMS" | tr -s ' ' | cut -d' ' -f5)
    echo $rms
}

function measure_peak() {
    ch_log=$(mktemp)
    ch $1 /dev/null 2>${ch_log}
    peak=$(cat $ch_log | grep "peak" | tr -s ' ' | cut -d' ' -f3)
    echo $peak
}

# Make power of a raw file $1 equal to the setpoint $2, buy adjusting the RMS level
function set_rms() {
    raw=$1
    setpoint_rms=$2

    rms=$(measure_rms $raw)
    gain=$(python3 -c "gain=${setpoint_rms}/${rms}; print(\"%f\" % gain)")
    raw_gain=$(mktemp)
    sox -t .s16 -r 8k -c 1 -v $gain $raw -t .s16 -r 8k -c 1 $raw_gain
    cp $raw_gain $raw
}

function spectrogram() {
    echo "pkg load signal; rx=load_raw(\"$1\"); plot_specgram(rx, Fs=8000, 0, 3000); print('-dpng',\"$2\"); quit" | octave-cli -qf
}
