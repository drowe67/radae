#!/bin/bash -x
#
# Generate test data on OFDM sync algorithm performance, used to plot BER v Eb/No curves

PATH=${PATH}:${PWD}
MODEL=model05/checkpoints/checkpoint_epoch_100.pth
# sample length drives length of simulation, good idea to have a longer sample for multipath tests
# to get closer to true mean
WAV=wav/all.wav

EbNodB_list='4 3 2 1 0 -1 -2 -3 -4 -5 -6 -6 -8'
#EbNodB_list='4'

function run_curve {
    results=$1
    shift
    log=$(mktemp)
    rm -f ${results}

    for EbNodB in $EbNodB_list
    do
        inference.sh ${MODEL} ${WAV} /dev/null --EbNodB ${EbNodB} --ber_test "$@" > ${log}
        ber=$(cat $log | grep "BER:" | tr -s ' ' | cut -d' ' -f4)
        printf "%5.2f %5.3f\n" $EbNodB $ber >> $results
    done
}

if [ $1 == "multipath" ]; then
    # generate multipath .f32 files (if they don't already exist)
    if [ ! -f os_h_mpp.f32 ] || [ ! -f os_g_mpp.f32 ]; then
        echo "Fs=8000; Rs=50; Nc=20; multipath_samples(\"mpp\", Fs, Rs, Nc, 120, \"os_h_mpp.f32\",\"os_g_mpp.f32\")" | octave-cli -qf
    fi
    # Multipath tests
    run_curve ofdm_sync_mp.txt --h_file os_h_mpp.f32
    run_curve ofdm_sync_mp_eq_ls.txt    --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --g_file os_g_mpp.f32
    #run_curve ofdm_sync_mp_eq_ls_f2.txt --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --g_file os_g_mpp.f32 --freq_offset 2 --gain 0.1
else
    # AWGN tests
    run_curve ofdm_sync.txt                --rate_Fs --pilots
    run_curve ofdm_sync_pilot_eq.txt       --rate_Fs --pilots --pilot_eq
    run_curve ofdm_sync_pilot_eq_f2.txt    --rate_Fs --pilots --pilot_eq --freq_offset 2
    run_curve ofdm_sync_pilot_eq_g0.1.txt  --rate_Fs --pilots --pilot_eq --gain 0.1
    run_curve ofdm_sync_pilot_eq_ls.txt    --rate_Fs --pilots --pilot_eq --eq_ls
    run_curve ofdm_sync_pilot_eq_ls_f2.txt --rate_Fs --pilots --pilot_eq --eq_ls --freq_offset 2 --gain 0.1
fi

