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
    echo "  usage ./ota_test.sh ~/.cache/LibriSpeech/test-clean  ~/.cache/LibriSpeech/test-awgn-2dB --awgn 2"
    echo
    echo "    --awgn SNRdB              AWGN channel simulation"
    echo "    -n numSamples             number of dataset samples to process (default all)"
    echo "    -d                        verbose debug information"
    exit
}

n_samples=0

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --awgn)
        awgn_snr_dB="$2"
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

# process audio files and place in new dataset directory
function process {
    pushd $source; flac=$(find . -name '*.flac'); popd
    if [ $n_samples -ne 0 ]; then
        flac=$(echo "$flac" | head -n $n_samples)
    fi

    n=$(echo "$flac" | wc -l)
    printf "Processing %d samples in dataset\n" $n

    for f in $flac
    do
        d=$(dirname $f)
        mkdir -p ${dest}/${d}
        sox ${source}/${f} -t .s16 -r 8000 - | ch - - --No -30 | sox -t .s16 -r 8000 -c 1 - -r 16000 ${dest}/${f}
    done
}

#cp_translation_files
process

