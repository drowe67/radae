#!/usr/bin/env bash
# asr_test_awgn.sh
#
# Top level ASR test script for AWGN and MPP channels

n=500

function print_help {
    echo
    echo "Generate curve data for ASR Radio Autoencoder testing"
    echo
    echo "  usage ./asr_test_top.sh rade|bbfm  [test option below]"
    echo "  usage ./asr_test.sh rade"
    echo "  usage ./asr_test.sh bbfm -n 100"
    echo
    echo "    -n numSamples             number of dataset samples to process (default 500)"
    echo "    --results resultsFile     name of results file (default results.txt)"
    echo "    -d                        verbose debug information"
    exit
}

function ssb {
    local results_file=$1
    No_range=$2
    for No in $No_range
    do
        ./asr_test.sh ssb --No $No -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep ssb | sed -e "s/ssb//" > tmp.txt 
    mv tmp.txt ${results_file}
}

function rade {
    local results_file=$1
    EbNodB_range=$2
    for EbNodB in $EbNodB_range
    do
        ./asr_test.sh rade --EbNodB $EbNodB -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep rade | sed -e "s/rade//" > tmp.txt
    mv tmp.txt ${results_file}
}

function freedv_700D {
    local results_file=$1
    No_range=$2
    for No in $No_range
    do
        ./asr_test.sh 700D --No $No -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep 700D | sed -e "s/700D//" > tmp.txt 
    mv tmp.txt ${results_file}
}

function fm {
    local results_file=$1
    RdBm_range=$2
    for RdBm in $RdBm_range
    do
        ./asr_test.sh fm --RdBm $RdBm -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep fm | sed -e "s/fm//" > tmp.txt
    mv tmp.txt ${results_file}
}

function bbfm {
    local results_file=$1
    RdBm_range=$2
    for RdBm in $RdBm_range
    do
        ./asr_test.sh bbfm --RdBm $RdBm -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep bbfm | sed -e "s/bbfm//" > tmp.txt
    mv tmp.txt ${results_file}
}

# Curves for 2024 HF RADE paper
function rade_hf_top {
    freedv_700D ${results_file}_awgn_700D.txt  "-100 -30 -26 -23 -20 -17 -15 -13"
    freedv_700D ${results_file}_mpp_700D.txt   "-100 -39 -36 -33 -30 -27" "--g_file g_mpp.f32"
    #freedv_700D ${results_file}_awgn_700D.txt  "-100 -38 -35 -32 -29 -26 -23 -20 -17"
    #freedv_700D ${results_file}_mpp_700D.txt   "-100 -44 -39 -36 -33 -30 -27" "--g_file g_mpp.f32"
    exit 0

    # run the controls
    controls_file=${results_file}_controls.txt
    rm -f ${controls_file}
    ./asr_test.sh clean -n $n --results ${controls_file}
    ./asr_test.sh fargan -n $n --results ${controls_file}
    ./asr_test.sh 4kHz -n $n --results ${controls_file}
    ./asr_test.sh ssb -n $n --results ${controls_file}
    ./asr_test.sh rade -n $n --results ${controls_file}
    # strip off all but last column for Octave plotting
    cat ${controls_file} | awk '{print $NF}' > ${results_file}_c.txt


    ssb  ${results_file}_awgn_ssb.txt  "-100 -38 -35 -32 -29 -26 -23 -20 -17"
    rade ${results_file}_awgn_rade.txt "100 15 10 5 2.5 0 -2.5"
    ssb  ${results_file}_mpp_ssb.txt   "-100 -44 -39 -36 -33 -30 -27" "--g_file g_mpp.f32"
    rade ${results_file}_mpp_rade.txt  "100 15 10 5 2.5 0" "--g_file g_mpp.f32"
}

# Curves for 2025 LMR paper
function bbfm_top {
    # run the single point controls
    controls_file=${results_file}_controls.txt
    rm -f ${controls_file}
    ./asr_test.sh clean -n $n --results ${controls_file}
    # use --sil 0 as FARGAN WER was high without it, could hear an artefact in the
    # intersample silence section at the end of some output files.  OK to remove
    # intersample silence as very little processing delay with FARGAN alone
    ./asr_test.sh fargan -n $n --results ${controls_file} --sil 0
    # strip off all but last column for Octave plotting
    cat ${controls_file} | awk '{print $NF}' > ${results_file}_c.txt

    fm ${results_file}_awgn_fm.txt "-100 -110 -115 -118 -120 -122 -125"
    bbfm ${results_file}_awgn_bbfm.txt "-100 -110 -120 -125 -126 -127"
    fm ${results_file}_lmr60_fm.txt "-100 -105 -110 -113 -115 -117 -120" "--h_file h_lmr60_Fs_8000Hz.f32"
    bbfm ${results_file}_lmr60_bbfm.txt "-100 -110 -120 -122 -123 -125 -126" "--h_file h_lmr60_Rs_2000Hz.f32"
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -n)
        n="$2"
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
if [ $mode == 'rade' ]; then
  results_file=241221_asr
  rade_hf_top
  exit 0
fi
if [ $mode == 'bbfm' ]; then
  results_file=250807_asr
  bbfm_top
  exit 0
fi

print_help

