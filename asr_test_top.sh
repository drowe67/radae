#!/usr/bin/env bash
# asr_test_awgn.sh
#
# Top level ASR test script for AWGN and MPP channels
set -x
results_file=241219_asr
n=100

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

#ssb  ${results_file}_awgn_ssb.txt  "-100 -39"
#ssb  ${results_file}_mpp_ssb.txt   "-100 -44" "--g_file g_mpp.f32"

ssb  ${results_file}_awgn_ssb.txt  "-100 -39 -36 -33 -30 -25 -20"
rade ${results_file}_awgn_rade.txt "100 15 10 5 2.5 0 -2.5"
ssb  ${results_file}_mpp_ssb.txt   "-100 -44 -41 -39 -36 -33 -30" "--g_file g_mpp.f32"
rade ${results_file}_mpp_rade.txt  "100 15 10 5 2.5 0" "--g_file g_mpp.f32"
