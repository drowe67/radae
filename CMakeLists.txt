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

# Build opus with FARGAN support.
include(cmake/BuildOpus.cmake)

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
add_test(NAME ota_test_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -21 0.3")

# Low SNR ota_test.sh, with chirp measurement, MPP, high loss threshold as we only care about gross errors,
# like stuck in false sync
add_test(NAME ota_test_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -24 0.4 --mpp --freq -25")


# Acquisition tests ------------------------------------------------------------------------------------

# noise-only test, should not acquire for 120s (currently set at 30s as it's too slow)
add_test(NAME acq_noise
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       dd if=/dev/zero of=/dev/stdout bs=16000 count=30 | \
                       ${CODEC2_DEV_BUILD_DIR}/src/ch - rx.int16 --No -20; \          # real int16 output
                       cat rx.int16 | python3 int16tof32.py --zeropad > rx.f32; \     # ..IQIQI.. .f32 with Q == 0
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16")
                       set_tests_properties(acq_noise PROPERTIES PASS_REGULAR_EXPRESSION "Acquisition failed")

# sine wave + noise-only test, should not acquire 
add_test(NAME acq_sine
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CODEC2_DEV_BUILD_DIR}/misc/mksine - 1000 30 | \
                       ${CODEC2_DEV_BUILD_DIR}/src/ch - rx.int16 --No -20; \          # real int16 output
                       cat rx.int16 | python3 int16tof32.py --zeropad > rx.f32; \     # ..IQIQI.. .f32 with Q == 0
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16")
                       set_tests_properties(acq_sine PROPERTIES PASS_REGULAR_EXPRESSION "Acquisition failed")

# Worst case: 0dB SNR MPP
add_test(NAME acq_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; F_OFF=10; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 3 --freq_offset $F_OFF --g_file g_mpp.f32 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq \
                       --bottleneck 3 --cp 0.004 --acq_test --fmax_target $F_OFF --acq_time_target 1.5")
                       set_tests_properties(acq_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# 0dB SNR MPG (slow fading to simulate quasi-stationary notches)
add_test(NAME acq_mpg
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; F_OFF=40; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 3 --freq_offset $F_OFF --g_file g_mpg.f32 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq \
                       --bottleneck 3 --cp 0.004 --acq_test --fmax_target $F_OFF --acq_time_target 1.5")
                       set_tests_properties(acq_mpg PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# 3dB SNR MPD (very fast fading, we expect reduced perf, so raise SNR)
add_test(NAME acq_mpd
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; F_OFF=-34.5; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 6 --freq_offset $F_OFF --g_file g_mpd.f32 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq \
                       --bottleneck 3 --cp 0.004 --acq_test --fmax_target $F_OFF --acq_time_target 1.5")
                       set_tests_properties(acq_mpd PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Acquisition test as above with an interfering sine wave at -3dBc on our signal, we relax acq time target to 2s
add_test(NAME acq_sine_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 3 --freq_offset -40 --g_file g_mpp.f32 --sine_amp 0.5 --sine_freq 1250 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq \
                       --bottleneck 3 --cp 0.004 --acq_test --fmax_target -40 --acq_time_target 2.0")
                       set_tests_properties(acq_sine_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

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

# basic test of streaming rx, run rx in vanilla and streaming , compare
add_test(NAME radae_rx_basic
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Introduce noise-only segments at either end to give state machine/end of over a work out.  Run for 50 seconds or so
# to test we don't drop sync in these poor channels (loss.py will pick up any interruptions)
                       
# SNR=-2dB AWGN
add_test(NAME radae_rx_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPP
# End of over detection is not always reliable on MPP, so run-on timer terminates over, --clip_end removes garbage at end
# We don't don't bother checking acquisition time on this channel, as it's a severe case.
add_test(NAME radae_rx_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPG
add_test(NAME radae_rx_mpg
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpg.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpg PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=3dB MPD, nasty channel that is fast fading but generally high SNR - so mission here is "don't break" rather than low SNR
add_test(NAME radae_rx_mpd
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 6 --freq_offset -28 --g_file g_mpd.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpd PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=-2dB AWGN ~5 Hz/min = 5/60 Hz/s freq drift
add_test(NAME radae_rx_dfdt
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                      ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --df_dt 0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.8")
                       set_tests_properties(radae_rx_dfdt PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=7dB ability to handle small differences in sample rate between tx and rx (delta Fs)
add_test(NAME radae_rx_dfs
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8001 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_dfs PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock > tx ADC clock.  We make sample clock error larger than 200ppm spec
# in order to exercise code.  Nice thing about "nin" design is it allows us to get meaningful "loss.py" results, ie no frames are lost.
add_test(NAME radae_rx_slip_plus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.08; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8005 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 1.0")
                       set_tests_properties(radae_rx_slip_plus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock < tx ADC clock
add_test(NAME radae_rx_slip_minus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 31 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.11; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 7995 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 1.0")
                       set_tests_properties(radae_rx_slip_minus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

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

# Embedded data (--auxdata) use for false sync detection, we --clip_start as false sync messes up alignment of feat vecs, Eb/No adjusted
# to pass (but low SNR not really the aim of this test), --foff_err forces a false sync state after first sync
add_test(NAME radae_rx_mpp_aux
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --auxdata \
                       --EbNodB 4 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rx.py model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --foff_err -8 --auxdata > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_start 100")
                       set_tests_properties(radae_rx_mpp_aux PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# evaluate.sh -----------------------------------------------------------------------------------------------

# TODO test evaluate.sh to prevent bit rot
