cmake_minimum_required(VERSION 3.13)
project(RADAE
  VERSION 0.1
  DESCRIPTION "Radio Autoencoder - transmission of vocoder features over radio channels"
  HOMEPAGE_URL "https://github.com/drowe67/radae"
  LANGUAGES C
  )

option(AVX "Enable AVX CPU optimizations." ON)

if(NOT CODEC2_DEV_BUILD_DIR)
    set(CODEC2_DEV_BUILD_DIR $HOME/codec2-dev/build_linux )
endif()

# Find or build opus as needed
if(NOT OPUS_DIR)
    set(OPUS_DIR $ENV{HOME}/opus)
endif()
include(cmake/FindOrBuildOpus.cmake)

#
# Prevent in-source builds
# If an in-source build is attempted, you will still need to clean up a few
# files manually.
#
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds in ${CMAKE_BINARY_DIR} are not "
   "allowed, please remove ./CMakeCache.txt and ./CMakeFiles/, create a "
   "separate build directory and run cmake from there.")
endif("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")

# Set default flags (from opus-ng build)
set(CMAKE_C_FLAGS "-O2 -fvisibility=hidden -fstack-protector-strong -W -Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes")

add_subdirectory(src)

# Ctests ----------------------------------------------------------------------

include(CTest)
enable_testing()

# Basic test of radae/radae.py code at rate Rs - pass condition is simply that it runs
add_test(NAME inference_model5
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB 10")

# Vanilla tests of model 17 & 18, just to confirm they're working
add_test(NAME inference_model17
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                        --EbNodB 0 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3")
add_test(NAME inference_model18
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        ./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --latent-dim 40 \
                        --EbNodB 0 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3")

# Stateful encoder sanity test (no channel noise)
add_test(NAME stateful_encoder
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; ./stateful_encoder.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --loss_test 0.2")
         set_tests_properties(stateful_encoder PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Stateful decoder sanity test (no channel noise)
add_test(NAME stateful_decoder
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; ./stateful_decoder.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --loss_test 0.2")
         set_tests_properties(stateful_decoder PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# DIGITAL SYMBOL BER TESTS ----------------------------------------------------------

# Substitute digital QPSK symbols and check BER; tests rate Fs subsystem, pilot and CP insertion, eq_ls

# Vanilla high SNR test
add_test(NAME inference_ber
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots \
                        --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test")
         set_tests_properties(inference_ber PROPERTIES PASS_REGULAR_EXPRESSION "BER: 0.000")

# As above but on AWGN at operating point Eb/No - gives LS EQ a work out
add_test(NAME inference_ber_awgn
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/inference_ber_awgn.sh")
         set_tests_properties(inference_ber_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# As above but on MPP at operating point Eb/No
add_test(NAME inference_ber_mpp
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/inference_ber_mpp.sh")
         set_tests_properties(inference_ber_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# ML SYMBOL OP POINT LOSS TESTS ----------------------------------------------------------

# all adjusted for -2.5dB-ish SNR operating point AWGN, should be loss < 0.3

add_test(NAME inference_loss_model5
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB -2.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model5 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME inference_loss_model17
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 0.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model17 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME inference_loss_model18
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --latent-dim 40 \
                       --EbNodB 3.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model18 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# rx.sh/ota_test.sh tests ------------------------------------------------------------------------------------

# Generate rx.f32 rate Fs IQ samples, run through stand alone rx/py receiver, measure loss and acquisition time
add_test(NAME rx_loss_acq_time
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --EbNodB 3  --freq_offset 10 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --correct_freq_offset --write_rx rx.f32; \
                       rm -f features_rx_out.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss_test 0.3 --acq_time_test 0.8")
                       set_tests_properties(rx_loss_acq_time PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Estimating C/No on multipath channels using a chirp
add_test(NAME chirp_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/chirp_mpp.sh ${CODEC2_DEV_BUILD_DIR} -16")
                       set_tests_properties(chirp_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Low SNR ota_test.sh, with chirp measurement, AWGN
# FIXME: this will pass if either condition is true
add_test(NAME ota_test_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -21")
                       set_tests_properties(ota_test_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
                       set_tests_properties(ota_test_awgn PROPERTIES PASS_REGULAR_EXPRESSION "CNodB PASS")


# FIXME: this will pass if either condition is true
# Low SNR ota_test.sh, with chirp measurement, MPP
add_test(NAME ota_test_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -25 --mpp --freq -25")
                       set_tests_properties(ota_test_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
                       set_tests_properties(ota_test_mpp PROPERTIES PASS_REGULAR_EXPRESSION "CNodB PASS")


# Acquisition tests ------------------------------------------------------------------------------------

# noise-only test, should not acquire for 120s (currently set at 30s as it's too slow)
add_test(NAME acq_noise
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       dd if=/dev/zero of=/dev/stdout bs=16000 count=30 | \
                       ${CODEC2_DEV_BUILD_DIR}/src/ch - rx.int16 --No -20; \  # real int16 output
                       cat rx.int16 | python3 int16tof32.py --zeropad > rx.f32; \     # ..IQIQI.. .f32 with Q == 0
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16")
                       set_tests_properties(acq_noise PROPERTIES PASS_REGULAR_EXPRESSION "Acquisition failed")

# Worst case: 0dB SNR MPP, make sure mean acq time < 1s
add_test(NAME acq_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g_mpp.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 3 --freq_offset 10 --g_file g_mpp.f32 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq \
                       --bottleneck 3 --cp 0.004 --acq_test --fmax_target 10")
                       set_tests_properties(acq_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# TODO: Add (a) interfering sine wave (b) stationary notch or perhaps slow fading

# Streaming Tx and Rx --------------------------------------------------------------------------------------

# basic test of streaming Tx/Rx, compare to vanilla Tx in inference.py
add_test(NAME radae_tx_basic
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat features_in.f32 | python3 radae_tx.py model17/checkpoints/checkpoint_epoch_100.pth > rx.f32
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_txs_out.f32; \
                       python3 loss.py features_in.f32 features_txs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_tx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# complex bandpass filter
add_test(NAME complex_bpf
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; python3 -c 'from radae import complex_bpf_test; complex_bpf_test(0)'")
                set_tests_properties(complex_bpf PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# compare rx.py in with vanilla and stateful core decoder, tests just ML part of streaming receiver
add_test(NAME rx_stateful
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 0 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16 --stateful;
                       cp features_rx_out.f32 features_rx_stateful_out.f32;
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16;
                       python3 loss.py features_in.f32 features_rx_out.f32 --features_hat2 features_rx_stateful_out.f32 --compare;")
                       set_tests_properties(rx_stateful PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# compare rx.py in vanilla and streaming mode, tests streaming receiver DSP and ML
add_test(NAME rx_streaming
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                        --EbNodB 0 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --time_offset -16  --coarse_mag --rx_one;
                       cp features_rx_out.f32 features_rx_one_out.f32;
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --time_offset -16 --coarse_mag;
                       python3 loss.py features_in.f32 features_rx_out.f32 --features_hat2 features_rx_one_out.f32 --compare;")
                       set_tests_properties(rx_streaming PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# basic test of streaming rx code, run rx.py in vanilla and streaming mode, compare against non-streaming rx.sh
add_test(NAME radae_rx_basic
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# introduce noise-only segments at either end to give state machine/end of over a work out
                       
# low SNR AWGN
add_test(NAME radae_rx_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Low SNR MPP
# End of over detection is not reliable on MPP, so run-on timer terminates over, --clip_end removes garbage at end
# We don't don't bother checking acquisition time on this channel, as it's a severe case
add_test(NAME radae_rx_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g_mpp.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 4 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Low SNR AWGN ~5 Hz/min = 5/60 Hz/s freq drift
add_test(NAME radae_rx_dfdt
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                      ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --df_dt 0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.8")
                       set_tests_properties(radae_rx_dfdt PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# ability to handle small differences in sample rate between tx and rx (delta Fs)
add_test(NAME radae_rx_dfs
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8001 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_dfs PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# profiles a run with a 50 second file (no pass/fail, run with -V to get a rough idea of execution time)
add_test(NAME radae_rx_profile
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --df_dt 0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 -m cProfile -s time radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 0 --no_stdout | head -n20")

# performs a run using the streaming FARGAN decoder, ie the full simplex rx decode stack.  No pass/fail, just for characterisation of run time
add_test(NAME radae_rx_fargan
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth $WAV /dev/null \
                       --EbNodB 10 --freq_offset 13 --df_dt -0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       \\time -o log.txt -f '%e' cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 0 | \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -fargan-synthesis - /dev/null; \
                       RUN_TIME=$(cat log.txt); DUR=$(soxi -D $WAV); percent=$(python3 -c \"percent=100*$RUN_TIME/$DUR; print('%f' % percent)\"); \
                       printf \"\nrun time: %5.2f duration: %5.2f percent CPU: %5.2f\n\n\" $RUN_TIME $DUR $percent ")


# evaluate.sh -----------------------------------------------------------------------------------------------

# TODO test evaluate.sh to prevent bit rot
