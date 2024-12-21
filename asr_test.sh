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
    echo "  usage ./asr_test.sh ssb|rade|fargan|4kHz [test option below]"
    echo "  usage ./ota_test.sh ssb --No -30"
    echo "  usage ./ota_test.sh rade --EbNodB 10"
    echo
    echo "    --EbNodB EbNodB           inference.py simulation noise level (experiment to get desired SNR)"
    echo "    --No NodB                 ch channel simulation No value (experiment to get desired SNR)"
    echo "    -n numSamples             number of dataset samples to process (default all)"
    echo "    --results resultsFile     name of results file (deafult results.txt)"
    echo "    -d                        verbose debug information"
    exit
}

n_samples=0
No=-100
EbNodB=100
setpoint_rms=2048
comp_gain=6
results=asr_results.txt
inference_args=""
ch_args=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --EbNodB)
        EbNodB="$2"
        shift
        shift
    ;;
    --g_file)
        g_file="$2"
        if [ ! -f $2 ]; then
            echo "can't find $2"
            exit 1
        fi
        inference_args="${inference_args} --g_file ${2}"
        cp ${2} fast_fading_samples.float	
        ch_args="${ch_args} --fading_dir . --mpp --gain 0.5"
        shift
        shift
    ;;
    --No)
        No="$2"
        shift
        shift
    ;;
    --results)
        results="$2"
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

if [ $# -lt 1 ]; then
    print_help
fi
mode=$1

source=~/.cache/LibriSpeech/test-clean
if [ ! -d $source ]; then
  echo "cant find Librispeech source directory" $source
  exit 1 
fi
# results must be written to a directory known by Librispeech package (can't be any name)
dest=~/.cache/LibriSpeech/test-other
rm -Rf $dest

# cp translation files to new dataset directory
function cp_translation_files {
    pushd $source > /dev/null; trans=$(find . -name '*.txt'); popd  > /dev/null
    for f in $trans
    do
        d=$(dirname $f)
        mkdir -p ${dest}/${d}
        cp ${source}/${f} ${dest}/${f}
    done
}

function print_mean_text_file {
    file_name=$1
    python3 - <<END
import numpy as np
s=np.loadtxt("${file_name}")
print(f"{np.mean(s):5.2f}", end='')
END
}

# process audio files and place in new dataset directory
function process {
    pushd $source > /dev/null; flac=$(find . -name '*.flac'); popd > /dev/null
    if [ $n_samples -ne 0 ]; then
        flac=$(echo "$flac" | head -n $n_samples)
    fi

    n=$(echo "$flac" | wc -l)
    printf "Processing %d samples in dataset\n" $n

    in=in.raw
    comp=comp.raw
    ch_log=ch_log.txt
    rade_log=rade_log.txt
    snr_log=snr_log.txt
    asr_log=asr.txt
    rm -f ${snr_log}
    CNo_log=CNo_log.txt
    rm -f ${CNo_log}

    if [ $mode == "ssb" ] || [ $mode == "4kHz" ]; then
        
        fading_adv=0
        for f in $flac
        do
            d=$(dirname $f)
            mkdir -p ${dest}/${d}

            if [ $mode == "ssb" ]; then
                sox ${source}/${f} -t .s16 -r 8000 ${in}
                # AGC and Hilbert compression
                set_rms ${in} $setpoint_rms
                analog_compressor  ${in} ${comp} ${comp_gain} 2>/dev/null
                ch ${comp} - --No ${No} ${ch_args} --fading_adv ${fading_adv} 2>${ch_log} | sox -t .s16 -r 8000 -c 1 - -r 16000 ${dest}/${f}
                grep "Fading file finished" $ch_log
                if [ $? -eq 0 ]; then
                    echo "Error - fading file too short after" $fading_adv " seconds"
                    exit 1
                fi
                snr=$(cat $ch_log | grep "SNR3k" | tr -s ' ' | cut -d' ' -f3)
                CNo=$(cat $ch_log | grep "SNR3k" | tr -s ' ' | cut -d' ' -f5)
                echo $snr >> ${snr_log}
                echo $CNo >> ${CNo_log}

                # advance through fading simulation file
                dur=$(sox --info -D ${source}/${f})
                fading_adv=$(python3 -c "print(${fading_adv} + ${dur})")
            else
              # $mode == "4kHz" (4kHz bandwidth, representing ideal Fs=8kHz vocoder)
              sox ${source}/${f} -r 8000 -t .s16 -c 1 - | sox -r 8000 -t .s16 -c 1 - -r 16000 ${dest}/${f}
            fi
        done
        if [ $mode == "ssb" ]; then
          SNR_mean=$(print_mean_text_file ${snr_log})
          CNo_mean=$(print_mean_text_file ${CNo_log})
        fi
    fi
    
    if [ $mode == "rade" ] || [ $mode == "fargan" ]; then
        # find length of each file
        duration_log=""
        flac_full=""
        pushd $source > /dev/null;
        for f in $flac
        do
          duration_log+=$(sox --info -D ${f})
          duration_log+=" "
          flac_full+="${source}/${f} /tmp/silence.wav "
        done
        popd > /dev/null; 
        
        # cat samples into one long input file, insert 500ms at end of sample to allow for processing at output
        sil=0.5
        sox -n -r 16000 -c 1 /tmp/silence.wav trim 0.0 ${sil}
        sox $flac_full -t .s16 ${in}

        # process all samples as one file to save time

        if [ $mode == "rade" ]; then
            ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth ${in} out.wav \
            --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata  --time_offset -16 \
            --EbNodB $EbNodB ${inference_args} | tee ${rade_log}
            grep "Multipath Doppler spread file too short" $rade_log
            if [ $? -eq 0 ]; then
                echo "Error - fading file too short"
                exit 1
            fi

            SNR_mean=$(cat $rade_log | grep "Measured" | tr -s ' ' | cut -d' ' -f4)
            CNo_mean=$(cat $rade_log | grep "Measured" | tr -s ' ' | cut -d' ' -f3)
        else
            # $mode == "fargan"
            ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth ${in} out.wav --auxdata --passthru
            #sox -t .s16 -r 16000 -c 1 ${in} out.wav
        fi

        # extract individual output files
        duration_array=( ${duration_log} )
        i=0
        st=0
        for f in $flac
        do
          dur=${duration_array[i]}
          dur=$(python3 -c "print($dur + ${sil})")
          #printf "%4d %s %5.2f %5.2f\n" $i $f $st $dur
          ((i++))
          if [ $i -eq ${#duration_array[@]} ]; then
            sox out.wav ${dest}/${f} trim $st
          else
            sox out.wav ${dest}/${f} trim $st $dur
          fi
          st=$(python3 -c "print($st + $dur)")
        done
    fi

    # test mode that just copies files
    if [ $mode == "clean" ]; then
        for f in $flac
        do
          cp ${source}/${f} ${dest}/${f}
        done
        
    fi

    python3 asr_wer.py test-other -n $n_samples --model turbo | tee > $asr_log
    wer=$(tail -n1 $asr_log | tr -s ' ' | cut -d' ' -f2)
    if [ $mode == "ssb" ] || [ $mode == "rade" ]; then
      printf "%-4s %5.2f %5.2f %5.2f\n" $mode $SNR_mean $CNo_mean $wer | tee -a $results
    else
      printf "%-4s %5.2f\n" $mode $wer | tee -a $results
    fi
}

cp_translation_files
process

