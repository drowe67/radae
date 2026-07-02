#!/usr/bin/env bash
# asr_test_awgn.sh
#
# Top level ASR test script for AWGN and MPP channels
set -x
results_file=241221_asr
n=500
mode=${1:-all}
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        -n) n=$2; shift 2;;
        *) shift;;
    esac
done

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

function radev2 {
    local results_file=$1
    EbNodB_range=$2
    for EbNodB in $EbNodB_range
    do
        ./asr_test.sh radev2 --EbNodB $EbNodB -n $n --results ${results_file} $3
    done
    cat ${results_file} | grep radev2 | sed -e "s/radev2//" > tmp.txt
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

if [ "$mode" == "700D" ] || [ "$mode" == "all" ]; then
    freedv_700D ${results_file}_awgn_700D.txt  "-100 -30 -26 -23 -20 -17 -15 -13"
    freedv_700D ${results_file}_mpp_700D.txt   "-100 -39 -36 -33 -30 -27" "--g_file g_mpp.f32"
fi

if [ "$mode" == "controls" ] || [ "$mode" == "all" ]; then
    controls_file=${results_file}_controls.txt
    rm -f ${controls_file}
    ./asr_test.sh clean -n $n --results ${controls_file}
    ./asr_test.sh fargan -n $n --results ${controls_file}
    ./asr_test.sh 4kHz -n $n --results ${controls_file}
    # strip off all but last column for Octave plotting
    cat ${controls_file} | awk '{print $NF}' > ${results_file}_c.txt
fi

if [ "$mode" == "ssb" ] || [ "$mode" == "all" ]; then
    ssb    ${results_file}_awgn_ssb.txt    "-100 -38 -35 -32 -29 -26 -23 -20 -17"
    ssb    ${results_file}_mpp_ssb.txt    "-100 -44 -39 -36 -33 -30 -27" "--g_file g_mpp.f32"
fi

if [ "$mode" == "rade" ] || [ "$mode" == "all" ]; then
    rade   ${results_file}_awgn_rade.txt   "100 15 10 5 2.5 0 -2.5"
    rade   ${results_file}_mpp_rade.txt   "100 15 10 5 2.5 0" "--g_file g_mpp.f32"
fi

if [ "$mode" == "radev2" ] || [ "$mode" == "all" ]; then
    radev2 ${results_file}_awgn_radev2.txt "100 15 10 5 2.5 0 -2.5"
    radev2 ${results_file}_mpp_radev2.txt  "100 15 10 5 2.5 0" "--g_file g_mpp.f32"
fi

