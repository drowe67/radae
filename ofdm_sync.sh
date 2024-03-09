#!/bin/bash -x
#
# Generate test data on OFDM sync algorithms

PATH=${PATH}:${PWD}
MODEL=model05/checkpoints/checkpoint_epoch_100.pth
WAV=wav/peter.wav

EbNodB_list='4 3 2 1 0 -1 -2 -3 -4 -5 -6 -6 -8'

function run_curve {
    results=$1
    shift
    log=$(mktemp)
    dummy_wav=$(mktemp)
    rm -f ${results}

    for EbNodB in $EbNodB_list
    do
        inference.sh ${MODEL} ${WAV} ${dummy_wav} --EbNodB ${EbNodB} --rate_Fs --pilots --ber_test "$@" > ${log}
        ber=$(cat $log | grep "BER:" | tr -s ' ' | cut -d' ' -f4)
        printf "%5.2f %5.3f\n" $EbNodB $ber >> $results
    done
}

# TODO vanilla, different sync algorithms, freq offsets, gain offsets
run_curve ofdm_sync.txt
run_curve ofdm_sync_pilot_eq.txt --pilot_eq
run_curve ofdm_sync_pilot_eq_f2.txt --pilot_eq --freq_offset 2
run_curve ofdm_sync_pilot_eq_g0.1.txt --pilot_eq --gain 0.1
run_curve ofdm_sync_pilot_eq_ls.txt --pilot_eq --eq_ls
run_curve ofdm_sync_pilot_eq_ls_f2.txt --pilot_eq --eq_ls --freq_offset 2 --gain 0.1




