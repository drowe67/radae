#!/usr/bin/env bash
# asr_test_awgn.sh
#
# Top level ASR test script for AWGN channels

results_file=241217_asr_awgn
n=100

function ssb_awgn {
    No_range="-100 -39 -36 -33 -30 -25 -20"
    for No in $No_range
    do
        ./asr_test.sh ssb --No $No -n $n --results ${results_file}_ssb.txt
    done
    cat ${results_file}_ssb.txt | grep ssb | sed -e "s/ssb//" > tmp.txt
    mv tmp.txt ${results_file}_ssb.txt
}

function rade_awgn {
    EbNodB_range="100 15 10 5 2.5 0 -2.5"
    for EbNodB in $EbNodB_range
    do
        ./asr_test.sh rade --EbNodB $EbNodB -n $n --results ${results_file}_rade.txt
    done
    cat ${results_file}_rade.txt | grep rade | sed -e "s/rade//" > tmp.txt
    mv tmp.txt ${results_file}_rade.txt
}

ssb_awgn
rade_awgn
