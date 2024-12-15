#!/usr/bin/env bash
# asr_test.sh
#
# Automatic Speech Recognition (ASR) testing for the  Radio Autoencoder. This script
# takes the samples from a clean dataset (e.g. Librispeech test-clean), and generates
# a dataset with channel simulations (RADE, SSB etc) applied.

CODEC2_DEV=${CODEC2_DEV:-${HOME}/codec2-dev}
PATH=${PATH}:${CODEC2_DEV}/build_linux/src:${CODEC2_DEV}/build_linux/misc:${PWD}/build/src

which ch >/dev/null || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

source utils.sh

function print_help {
    echo
    echo "Automated Speech Recognition (ASR) dataset processing for Radio Autoencoder testing"
    echo
    echo "  usage ./asr_test.sh path/to/source dest [test option below]"
    echo "  usage ./ota_test.sh ~/.cache/LibriSpeech/test-clean  ~/.cache/LibriSpeech/test-awgn-2dB --No -30"
    echo
    echo "    --No NodB                 ch channel simulation No value (experiment to get desired SNR)"
    echo "    -n numSamples             number of dataset samples to process (default all)"
    echo "    -d                        verbose debug information"
    exit
}

n_samples=0
No=-100
setpoint_rms=2048
comp_gain=6
mode="rade_inf"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --No)
        No="$2"
        shift
        shift
    ;;
    -n)
        n_samples="$2"
        shift
        shift
    ;;
    -d)
        set -x;
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

if [ $# -lt 2 ]; then
    print_help
fi

source=$1
dest=$2
rm -Rf $dest

# cp translation files to new dataset directory
function cp_translation_files {
    pushd $source; trans=$(find . -name '*.txt'); popd
    for f in $trans
    do
        d=$(dirname $f)
        mkdir -p ${dest}/${d}
        cp ${source}/${f} ${dest}/${f}
    done
}

function mean_text_file {
    file_name=#1
    python3 <<EOF
    import numpy as np
    s=np.loadtxt(${file_name})
    print(f"{np.mean(s):5.2f}")
EOF
}

# process audio files and place in new dataset directory
function process {
    pushd $source; flac=$(find . -name '*.flac'); popd
    if [ $n_samples -ne 0 ]; then
        flac=$(echo "$flac" | head -n $n_samples)
    fi

    n=$(echo "$flac" | wc -l)
    printf "Processing %d samples in dataset\n" $n

    in=in.raw
    comp=comp.raw
    ch_log=ch_log.txt
    snr_log=snr_log.txt
    rm -f ${snr_log}

    if [ $mode == "ssb" ]; then
    
        for f in $flac
        do
            d=$(dirname $f)
            mkdir -p ${dest}/${d}
            sox ${source}/${f} -t .s16 -r 8000 ${in}
            # AGC and Hilbert compression
            set_rms ${in} $setpoint_rms
            analog_compressor  ${in} ${comp} ${comp_gain}
            papr=$(measure_cpapr ${comp})
            ch ${comp} - --No ${No}  2>${ch_log} | sox -t .s16 -r 8000 -c 1 - -r 16000 ${dest}/${f}
            snr=$(cat $ch_log | grep "SNR3k" | tr -s ' ' | cut -d' ' -f3)
            echo $snr >> ${snr_log}
            echo ${dest}/${f} ${snr} ${papr}
            print_mean_text_file ${snr_log}
        done
    fi        
    
    if [ $mode == "rade_inf" ]; then
        # find length of each file
        duration_log=""
        flac_full=""
        pushd $source;
        for f in $flac
        do
          duration_log+=$(sox --info -D ${f})
          duration_log+=" "
          flac_full+=${source}/${f}
          flac_full+=" "
        done
        popd; 
        
        # cat samples into one long input file
        sox $flac_full -t .s16 ${in}

        # process all samples as one file to save time
        ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth ${in} out.wav \
        --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata

        # extract individual output files
        duration_array=( ${duration_log} )
        i=0
        st=0
        for f in $flac
        do
          dur=${duration_array[i]}
          printf "%4d %s %5.2f %5.2f\n" $i $f $st $dur
          ((i++))
          if [ $i -eq ${#duration_array[@]} ]; then
            sox out.wav ${dest}/${f} trim $st
          else
            sox out.wav ${dest}/${f} trim $st $dur
          fi
          st=$(python3 -c "print($st + $dur)")
        done
    fi


}

cp_translation_files
process

