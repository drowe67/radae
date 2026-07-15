cmake_minimum_required(VERSION 3.13)
project(RADAE
  VERSION 0.1
  DESCRIPTION "Radio Autoencoder - transmission of vocoder features over radio channels"
  HOMEPAGE_URL "https://github.com/drowe67/radae"
  LANGUAGES C
  )

option(BUILD_SHARED_LIBS
    "Build shared library. Set to OFF for static library." ON)
option(AVX "Enable AVX CPU optimizations." ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.11" CACHE STRING "Minimum OS X deployment version" FORCE)

# Build universal ARM64 and x86_64 binaries on Mac.
if(BUILD_OSX_UNIVERSAL)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
endif(BUILD_OSX_UNIVERSAL)

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
set(CMAKE_C_FLAGS "-O2 -fstack-protector-strong -W -Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes -DHAVE_CONFIG_H")

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
                       --EbNodB 1  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model17 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME inference_loss_model18
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --latent-dim 40 \
                       --EbNodB 4  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
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
                       ./test/chirp_mpp.sh ${CMAKE_BINARY_DIR} -16")

# Low SNR ota_test.sh, with chirp measurement, AWGN
add_test(NAME ota_test_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ${CMAKE_BINARY_DIR} wav/brian_g8sez.wav -24 0.4")

# Low SNR ota_test.sh, with chirp measurement, MPP, high loss threshold as we only care about gross errors,
# like stuck in false sync
add_test(NAME ota_test_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ${CMAKE_BINARY_DIR} wav/brian_g8sez.wav -30 0.45 --mpp --freq -25")


# Acquisition tests ------------------------------------------------------------------------------------

# noise-only test, should not acquire for 120s (currently set at 30s as it's too slow)
add_test(NAME acq_noise
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       dd if=/dev/zero of=/dev/stdout bs=16000 count=30 | \
                       ${CMAKE_BINARY_DIR}/src/ch - rx.int16 --No -20; \          # real int16 output
                       cat rx.int16 | python3 int16tof32.py --zeropad > rx.f32; \     # ..IQIQI.. .f32 with Q == 0
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16")
                       set_tests_properties(acq_noise PROPERTIES PASS_REGULAR_EXPRESSION "Acquisition failed")

# sine wave + noise-only test, should not acquire 
add_test(NAME acq_sine
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_BINARY_DIR}/src/mksine - 1000 30 | \
                       ${CMAKE_BINARY_DIR}/src/ch - rx.int16 --No -20; \          # real int16 output
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
                       --EbNodB 4 --freq_offset $F_OFF --g_file g_mpp.f32 --write_rx rx.f32; \
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
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat features_in.f32 | python3 radae_txe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth --txbpf > rx.f32
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_txs_out.f32; \
                       python3 loss.py features_in.f32 features_txs_out.f32 --loss_test 0.15 --acq_time_test 0.5 --clip_start 5")
                       set_tests_properties(radae_tx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# complex bandpass filter
add_test(NAME complex_bpf
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; python3 -c 'from radae import complex_bpf_test; complex_bpf_test(0)'")
                set_tests_properties(complex_bpf PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# compare rx.py in with vanilla and stateful core decoder, tests just ML part of streaming receiver
add_test(NAME rx_stateful
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 6 --freq_offset 11 \
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
                        --EbNodB 6 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --time_offset -16  --coarse_mag --rx_one;
                       cp features_rx_out.f32 features_rx_one_out.f32;
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --time_offset -16 --coarse_mag;
                       python3 loss.py features_in.f32 features_rx_out.f32 --features_hat2 features_rx_one_out.f32 --compare;")
                       set_tests_properties(rx_streaming PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# basic test of streaming rx, run rx in vanilla and streaming, compare
add_test(NAME radae_rx_basic
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 --prepend_noise 1 --append_noise 1 --end_of_over --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | PYTHONPATH='../' python3 radae_rxe.py -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Introduce noise-only segments at either end to give state machine/end of over a work out.  Run for 50 seconds or so
# to test we don't drop sync in these poor channels (loss.py will pick up any interruptions)
                       
# SNR=-2dB AWGN
add_test(NAME radae_rx_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 1.0 --clip_end 300")
                       set_tests_properties(radae_rx_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR estimator spot tests: check radae_rxe.py reports SNR3k within 3 dB of inference.sh Measured SNR3k
add_test(NAME radae_snr_est_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 \
                       --correct_freq_offset --EbNodB 10 >snr_inf_high.txt 2>/dev/null; \
                       SNR3K=\$(grep Measured snr_inf_high.txt | awk '{print \$4}'); \
                       cat rx.f32 | python3 radae_rxe.py 2>&1 >/dev/null | python3 test/check_snr.py --snr3k \$SNR3K --tol 2.0")
         set_tests_properties(radae_snr_est_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME radae_snr_est_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 \
                       --correct_freq_offset --EbNodB 5 >snr_inf_low.txt 2>/dev/null; \
                       SNR3K=\$(grep Measured snr_inf_low.txt | awk '{print \$4}'); \
                       cat rx.f32 | python3 radae_rxe.py 2>&1 >/dev/null | python3 test/check_snr.py --snr3k \$SNR3K --tol 2.0")
         set_tests_properties(radae_snr_est_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPP
# End of over detection is not always reliable on MPP, so run-on timer terminates over, --clip_end removes garbage at end
# We don't don't bother checking acquisition time on this channel, as it's a severe case.
add_test(NAME radae_rx_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --disable_unsync 5.0 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.35 --clip_end 300")
                       set_tests_properties(radae_rx_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPG.  We disable_unsync as we don't really care if we re-sync on long fades - user won't care much.  Main aim is to make sure loss is OK
# in this channel
add_test(NAME radae_rx_mpg
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpg.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --disable_unsync 5.0 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpg PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=3dB MPD, nasty channel that is fast fading but generally high SNR - so mission here is "don't break" rather than low SNR
add_test(NAME radae_rx_mpd
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 6 --freq_offset -28 --g_file g_mpd.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(radae_rx_mpd PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=-2dB AWGN ~5 Hz/min = 5/60 Hz/s freq drift
add_test(NAME radae_rx_dfdt
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                      ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --df_dt 0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.8 --clip_end 100")
                       set_tests_properties(radae_rx_dfdt PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=7dB ability to handle small differences in sample rate between tx and rx (delta Fs)
add_test(NAME radae_rx_dfs
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8001 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_dfs PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock > tx ADC clock.  We make sample clock error larger than 200ppm spec
# in order to exercise code.  Nice thing about "nin" design is it allows us to get meaningful "loss.py" results, ie no frames are lost.  We're
# really just trying to exercise the slip code here, not too concerned about loss.  There should be a noticable transition in tmax in each test as
# it hits the upper or lower limit
add_test(NAME radae_rx_slip_plus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.06; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8005 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.2 --acq_time_test 1.0")
                       set_tests_properties(radae_rx_slip_plus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test bugfix for issue where a combination of timing slips and dropped frames over a long enough
# period of time results in the RADE DSP logic being unable to regain sync.
add_test(NAME radae_rx_slip_plus_drops
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        sox wav/long_qso.wav -t .s16 -r 8020 -c 1 - | \
                          python3 int16tof32.py --zeropad | \
                          OMP_NUM_THREADS=1 python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 2>&1 >/dev/null | \
                          tail -n1 | \
                          grep 'state: sync'")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock < tx ADC clock
add_test(NAME radae_rx_slip_minus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 31 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.11; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 7995 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.25 --acq_time_test 1.0")
                       set_tests_properties(radae_rx_slip_minus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# profiles a run with a 50 second file (no pass/fail, run with -V to get a rough idea of execution time)
add_test(NAME radae_rx_profile
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 -m cProfile -s time radae_rxe.py -v 1 --no_stdout | head -n20")

#  Characterise run time using full simplex Rx stack (Python core decoder version).  No pass/fail, just for characterisation of run time
add_test(NAME radae_rx_stack_py
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth $WAV /dev/null \
                       --EbNodB 10 --freq_offset 13 --df_dt -0.1 --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset ; \
                       \\time -o log.txt -f '%e' cat rx.f32 | python3 radae_rxe.py -v 0 | \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -fargan-synthesis - /dev/null; \
                       RUN_TIME=$(cat log.txt); DUR=$(soxi -D $WAV); percent=$(python3 -c \"percent=100*$RUN_TIME/$DUR; print('%f' % percent)\"); \
                       printf \"\nrun time: %5.2f duration: %5.2f percent CPU: %5.2f\n\n\" $RUN_TIME $DUR $percent ")

# Embedded data (--auxdata) use for false sync detection, we --clip_start as false sync messes up alignment of feat vecs, Eb/No adjusted
# to pass (but low SNR not really the aim of this test), --foff_err forces a false sync state after first sync
add_test(NAME radae_rx_aux_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --auxdata \
                       --EbNodB 4 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --foff_err 8 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_start 300")
                       set_tests_properties(radae_rx_aux_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")


# Test Embedded version of streaming Tx, running with Python __main__
add_test(NAME radae_tx_embed_py
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat features_in.f32 | python3 radae_txe.py > rx.f32; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 1 > features_txs_out.f32; \
                       python3 loss.py features_in.f32 features_txs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_tx_embed_py PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test Embedded version of streaming Rx, running with Python __main__
add_test(NAME radae_rx_embed_py
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 radae_rxe.py > features_out.f32; \
                       python3 loss.py features_in.f32 features_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_embed_py PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# EOO data -------------------------------------------------------------------------------------------

# Python Tx & Rx, no noise
add_test(NAME radae_eoo_data_py
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
                       cat features_in.f32 | python3 radae_txe.py --eoo_data_test > rx.f32; \
                       cat rx.f32 | python3 radae_rxe.py -v 2 --eoo_data_test > /dev/null")
                       set_tests_properties(radae_eoo_data_py PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# BBFM -----------------------------------------------------------------------------------------------

# single carrier modem internal (inside single_carrier class) tests
add_test(NAME bbfm_sc_internal
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       python3 -c 'from radae import single_carrier,single_carrier_tests; single_carrier_tests()'")
                       set_tests_properties(bbfm_sc_internal PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS")

# single carrier modem stand alone tx/rx, using BPSK symbol/BER test mode
add_test(NAME bbfm_sc_ber
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./bbfm_inference.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --write_latent z.f32; \
                       cat z.f32 | python3 sc_tx.py --ber_test > t.int16; \
                       cat t.int16 | python3 sc_rx.py --ber_test --target_ber 0 > /dev/null")
                       set_tests_properties(bbfm_sc_ber PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# single carrier modem stand alone tx/rx, using continuously valued BBFM symbols, compare loss in decoded features
add_test(NAME bbfm_sc_loss
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./bbfm_inference.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --write_latent z.f32; \
                       cat z.f32 | python3 sc_tx.py | python3 sc_rx.py > z_hat.f32; \
                       ./bbfm_rx.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 /dev/null; \
                       python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_rx_out.f32 --compare")
                       set_tests_properties(bbfm_sc_loss PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# single carrier modem stand alone tx/rx with external 300-2700Hz band pass filter and no noise, measure loss.  In practice, SNR will be
# quite high, so channel distortions other than noise may dominate
add_test(NAME bbfm_sc_bpf_loss
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./bbfm_inference.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --write_latent z.f32; \
                       cat z.f32 | python3 sc_tx.py | ${CMAKE_BINARY_DIR}/src/ch - - | python3 sc_rx.py > z_hat.f32; \
                       ./bbfm_rx.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 /dev/null; \
                       python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_rx_out.f32 --compare")
                       set_tests_properties(bbfm_sc_bpf_loss PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# RADAE_NOPY C Port tests (optional: only enabled when RADE_C_BUILD_DIR is set) ----------
if(RADE_C_BUILD_DIR)

# basic test of streaming Tx/Rx, compare to vanilla Tx in inference.py
add_test(NAME rade_c_v1_tx_basic
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat features_in.f32 | ${RADE_C_BUILD_DIR}/src/radae_tx > rx.f32
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_txs_out.f32; \
                       python3 loss.py features_in.f32 features_txs_out.f32 --loss_test 0.15 --acq_time_test 0.5 --clip_start 5")
                       set_tests_properties(rade_c_v1_tx_basic PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# TODO need unit test for band pass filter, Python test, C unit test

# SNR=-2dB AWGN
add_test(NAME rade_c_v1_rx_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 1.0 --clip_end 300")
                       set_tests_properties(rade_c_v1_rx_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPP
# End of over detection is not always reliable on MPP, so run-on timer terminates over, --clip_end removes garbage at end
# We don't don't bother checking acquisition time on this channel, as it's a severe case.
add_test(NAME rade_c_v1_rx_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --disable_unsync 5.0 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.35 --clip_end 300")
                       set_tests_properties(rade_c_v1_rx_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=0dB MPG.  We disable_unsync as we don't really care if we re-sync on long fades - user won't care much.  Main aim is to make sure loss is OK
# in this channel
add_test(NAME rade_c_v1_rx_mpg
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 3 --freq_offset -11 --g_file g_mpg.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --disable_unsync 5.0 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(rade_c_v1_rx_mpg PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=3dB MPD, nasty channel that is fast fading but generally high SNR - so mission here is "don't break" rather than low SNR
add_test(NAME rade_c_v1_rx_mpd
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 \
                       --EbNodB 6 --freq_offset -28 --g_file g_mpd.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata --correct_freq_offset; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --disable_unsync 5.0 > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --clip_end 300")
                       set_tests_properties(rade_c_v1_rx_mpd PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=-2dB AWGN ~5 Hz/min = 5/60 Hz/s freq drift
add_test(NAME rade_c_v1_rx_dfdt
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                      ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --df_dt 0.1 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --auxdata; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss 0.3 --acq_time_test 0.8 --clip_end 100")
                       set_tests_properties(rade_c_v1_rx_dfdt PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR=7dB ability to handle small differences in sample rate between tx and rx (delta Fs)
add_test(NAME rade_c_v1_rx_dfs
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8001 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(rade_c_v1_rx_dfs PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock > tx ADC clock.  We make sample clock error larger than 200ppm spec
# in order to exercise code.  Nice thing about "nin" design is it allows us to get meaningful "loss.py" results, ie no frames are lost.  We're
# really just trying to exercise the slip code here, not too concerned about loss.  There should be a noticable transition in tmax in each test as
# it hits the upper or lower limit
add_test(NAME rade_c_v1_rx_slip_plus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.06; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 8005 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.2 --acq_time_test 1.0")
                       set_tests_properties(rade_c_v1_rx_slip_plus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test bugfix for issue where a combination of timing slips and dropped frames over a long enough
# period of time results in the RADE DSP logic being unable to regain sync.
add_test(NAME rade_c_v1_rx_slip_plus_drops
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        sox wav/long_qso.wav -t .s16 -r 8020 -c 1 - | \
                          python3 int16tof32.py --zeropad | \
                          ${RADE_C_BUILD_DIR}/src/radae_rx 2>&1 >/dev/null | \
                          tail -n2 | \
                          grep 'state:       sync'")

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock < tx ADC clock
add_test(NAME rade_c_v1_rx_slip_minus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 31 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.11; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 7995 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.25 --acq_time_test 1.0")
                       set_tests_properties(rade_c_v1_rx_slip_minus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Characterise run time using full simplex Tx stack (C core encoder version).  No pass/fail, just for characterisation of run time
add_test(NAME rade_c_v1_tx_stack_c
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       \\time -o log.txt -f '%e' ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features $WAV - | 
                       ${RADE_C_BUILD_DIR}/src/radae_tx > /dev/null; \
                       RUN_TIME=$(cat log.txt); DUR=$(soxi -D $WAV); percent=$(python3 -c \"percent=100*$RUN_TIME/$DUR; print('%f' % percent)\"); \
                       printf \"\nrun time: %5.2f duration: %5.2f percent CPU: %5.2f\n\n\" $RUN_TIME $DUR $percent ")

# As above but with C core decoder for comparison
add_test(NAME rade_c_v1_rx_stack_c
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth $WAV /dev/null \
                       --EbNodB 10 --freq_offset 13 --df_dt -0.1 --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset ; \
                       \\time -o log.txt -f '%e' cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx | \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -fargan-synthesis - /dev/null; \
                       RUN_TIME=$(cat log.txt); DUR=$(soxi -D $WAV); percent=$(python3 -c \"percent=100*$RUN_TIME/$DUR; print('%f' % percent)\"); \
                       printf \"\nrun time: %5.2f duration: %5.2f percent CPU: %5.2f\n\n\" $RUN_TIME $DUR $percent ")

# EOO data: C tx & rx, no noise. Note Python radae_txe.py just generates eoo_tx.f32 for C radae_tx
add_test(NAME rade_c_v1_eoo_data_c
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
                       cat features_in.f32 | python3 radae_txe.py --eoo_data_test > /dev/null; \
                       cat features_in.f32 | ${RADE_C_BUILD_DIR}/src/radae_tx > rx.f32; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > /dev/null; \
                       python3 eoo_ber.py eoo_tx.f32 eoo_rx.f32")
                       set_tests_properties(rade_c_v1_eoo_data_c PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# C tx & rx, over a multipath channel, we set up 5 "overs", each with an EOO chunk of data.  Just one of them needs
# to have a BER < 5% for a PASS.  About 6dB SNR
add_test(NAME rade_c_v1_eoo_data_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/make_g.sh; \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
                       cat features_in.f32 | python3 radae_txe.py --eoo_data_test > /dev/null; \
                       cat features_in.f32 | ${RADE_C_BUILD_DIR}/src/radae_tx > tx.f32; \
                       cat tx.f32 tx.f32 tx.f32 tx.f32 tx.f32 > tx_2.f32;
                       cat tx_2.f32 | python3 f32toint16.py --real --scale 8192 > tx.raw; \
                       ${CMAKE_BINARY_DIR}/src/ch tx.raw rx.raw --No -24 --mpp --fading_dir .; \
                       cat rx.raw | python3 int16tof32.py --zeropad > rx.f32; \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx > /dev/null; \
                       python3 eoo_ber.py eoo_tx.f32 eoo_rx.f32")
                       set_tests_properties(rade_c_v1_eoo_data_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# BPF test - only one iteration
add_test(NAME rade_c_v1_bpf_singlepass
         COMMAND ${RADE_C_BUILD_DIR}/src/rade_bpf_test 1 8000)

# BPF test - filter in small chunks
add_test(NAME rade_c_v1_bpf_chunks
         COMMAND ${RADE_C_BUILD_DIR}/src/rade_bpf_test 10 960)

# SNR estimator spot tests: check C radae_rx reports SNR3k within 3 dB of inference.sh Measured SNR3k
add_test(NAME rade_c_v1_snr_est_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 \
                       --correct_freq_offset --EbNodB 10 >snr_inf_nopy_high.txt 2>/dev/null; \
                       SNR3K=\$(grep Measured snr_inf_nopy_high.txt | awk '{print \$4}'); \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx 2>&1 >/dev/null | python3 test/check_snr.py --snr3k \$SNR3K --tol 2.0")
         set_tests_properties(rade_c_v1_snr_est_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME rade_c_v1_snr_est_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 \
                       --correct_freq_offset --EbNodB 5 >snr_inf_nopy_low.txt 2>/dev/null; \
                       SNR3K=\$(grep Measured snr_inf_nopy_low.txt | awk '{print \$4}'); \
                       cat rx.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx 2>&1 >/dev/null | python3 test/check_snr.py --snr3k \$SNR3K --tol 2.0")
         set_tests_properties(rade_c_v1_snr_est_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# DIGITAL SYMBOL BER TESTS (C port) ------------------------------------------
# Substitute random QPSK symbols (+/-1) for encoder output, apply channel model,
# demodulate, and check BER against theoretical QPSK/MPP curves with 2 dB implementation loss.

# Noiseless: all bits must be correct
add_test(NAME rade_c_v1_ber
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                        ${RADE_C_BUILD_DIR}/src/rade_ber_test --EbNodB 100 --frames 100")
         set_tests_properties(rade_c_v1_ber PROPERTIES PASS_REGULAR_EXPRESSION "BER: 0.0000")

# AWGN: BER must be below theoretical QPSK target (with 2 dB implementation loss allowance)
add_test(NAME rade_c_v1_ber_awgn
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/rade_c_ber_awgn.sh ${RADE_C_BUILD_DIR}")
         set_tests_properties(rade_c_v1_ber_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# MPP: BER must be below theoretical two-path Rayleigh target (with 2 dB implementation loss allowance)
add_test(NAME rade_c_v1_ber_mpp
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/rade_c_ber_mpp.sh ${RADE_C_BUILD_DIR}")
         set_tests_properties(rade_c_v1_ber_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C port tests ----------------------------------------------------------------

# Test V2 C encoder: C encoder output fed to Python stateful decoder, check loss
add_test(NAME rade_c_v2_enc
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
             python3 -c \"import numpy as np; \
                 f=np.fromfile('features_in.f32',dtype=np.float32).reshape(-1,36); \
                 n=(len(f)//4)*4; \
                 f21=np.concatenate([f[:n,:20],-np.ones((n,1),dtype=np.float32)],axis=1); \
                 f21.flatten().tofile('features_v2_in.f32')\"; \
             cat features_v2_in.f32 | ${RADE_C_BUILD_DIR}/src/rade_enc_v2_test > z_v2_c.f32; \
             python3 stateful_encoder_v2.py 250725/checkpoints/checkpoint_epoch_200.pth \
                 features_in.f32 /dev/null --read_latent z_v2_c.f32 --loss_test 0.2")
         set_tests_properties(rade_c_v2_enc PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test V2 C decoder: Python stateful encoder latents fed to C decoder, compare with Python decoder
add_test(NAME rade_c_v2_dec
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
             python3 stateful_encoder_v2.py 250725/checkpoints/checkpoint_epoch_200.pth \
                 features_in.f32 features_v2_py.f32 --write_latent z_v2_py.f32; \
             cat z_v2_py.f32 | ${RADE_C_BUILD_DIR}/src/rade_dec_v2_test > features_v2_c_raw.f32; \
             python3 -c \"import numpy as np; \
                 f=np.fromfile('features_v2_c_raw.f32',dtype=np.float32).reshape(-1,21); \
                 out=np.concatenate([f[:,:20],np.zeros((len(f),16),dtype=np.float32)],axis=1); \
                 out.flatten().tofile('features_v2_c.f32')\"; \
             python3 loss.py features_in.f32 features_v2_py.f32 --features_hat2 features_v2_c.f32 --compare")
         set_tests_properties(rade_c_v2_dec PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test V2 C transmitter: compare IQ output to tx2.py reference using RMS difference
# EOO max diff threshold: 1e-4 (essentially float32 precision)
# Data diff_rms/sig_rms threshold: 3% (encoder float differences are ~1.2%)
add_test(NAME rade_c_v2_tx
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
             python3 tx2.py 250725/checkpoints/checkpoint_epoch_200.pth \
                 features_in.f32 tx2_py.f32; \
             cat features_in.f32 | ${RADE_C_BUILD_DIR}/src/radae_tx --v2 > tx2_c.f32; \
             python3 -c \"\
import numpy as np; \
py=np.fromfile('tx2_py.f32',dtype=np.complex64); \
c=np.fromfile('tx2_c.f32',dtype=np.complex64); \
n=min(len(py),len(c)); \
data_py=py[:n-960]; data_c=c[:n-960]; \
eoo_py=py[n-960:n]; eoo_c=c[n-960:n]; \
diff_rms=np.sqrt(np.mean(np.abs(data_py-data_c)**2)); \
sig_rms=np.sqrt(np.mean(np.abs(data_py)**2)); \
eoo_max=np.max(np.abs(eoo_py-eoo_c)); \
print(f'diff_rms/sig_rms: {diff_rms/sig_rms:.4f}  eoo_max_diff: {eoo_max:.2e}'); \
ok = diff_rms/sig_rms < 0.03 and eoo_max < 1e-4; \
print('PASS' if ok else 'FAIL')\"")
         set_tests_properties(rade_c_v2_tx PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test V2 C receiver: C tx -> C rx pipeline. At high SNR (no noise) the
# decoder output loss vs the original vocoder features should be <0.2.
add_test(NAME rade_c_v2_rx
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
             cat features_in.f32 | ${RADE_C_BUILD_DIR}/src/radae_tx --v2 > tx2_c.f32; \
             cat tx2_c.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 > features_rx_c.f32; \
             python3 loss.py features_in.f32 features_rx_c.f32 --loss_test 0.2 --clip_start 200")
         set_tests_properties(rade_c_v2_rx PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver AWGN high SNR
add_test(NAME rade_c_v2_rx_awgn
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
             --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
             --auxdata --w1_dec 128 --write_rx rx_v2_nopy.f32 \
             --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset; \
             cat rx_v2_nopy.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 > features_rx_v2_c.f32; \
             python3 loss.py features_in.f32 features_rx_v2_c.f32 --loss_test 0.15 \
             --acq_time_test 2.0 --clip_start 100 --clip_end 300")
         set_tests_properties(rade_c_v2_rx_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver AWGN low SNR: EbNodB=1
add_test(NAME rade_c_v2_rx_awgn_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
             --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
             --auxdata --w1_dec 128 --write_rx rx_v2_nopy.f32 \
             --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset \
             --EbNodB 1; \
             cat rx_v2_nopy.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 > features_rx_v2_c.f32; \
             python3 loss.py features_in.f32 features_rx_v2_c.f32 --loss_test 0.45 --clip_start 100 --clip_end 300")
         set_tests_properties(rade_c_v2_rx_awgn_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver MPP high SNR
add_test(NAME rade_c_v2_rx_mpp
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             test/make_g.sh; \
             ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
             --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
             --auxdata --w1_dec 128 --write_rx rx_v2_nopy.f32 \
             --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset \
             --g_file g_mpp.f32; \
             cat rx_v2_nopy.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 > features_rx_v2_c.f32; \
             python3 loss.py features_in.f32 features_rx_v2_c.f32 --loss_test 0.15 --clip_start 100 --clip_end 300")
         set_tests_properties(rade_c_v2_rx_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver MPP low SNR: EbNodB=6
add_test(NAME rade_c_v2_rx_mpp_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             test/make_g.sh; \
             ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
             --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
             --auxdata --w1_dec 128 --write_rx rx_v2_nopy.f32 \
             --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset \
             --g_file g_mpp.f32 --EbNodB 6; \
             cat rx_v2_nopy.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 > features_rx_v2_c.f32; \
             python3 loss.py features_in.f32 features_rx_v2_c.f32 --loss_test 0.40 --clip_start 100 --clip_end 300")
         set_tests_properties(rade_c_v2_rx_mpp_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver EOO detection: MPP channel at EbNodB=8 (mirrors v2_eoo_mpp_low)
add_test(NAME rade_c_v2_rx_eoo
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             test/make_g.sh; \
             ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null \
             --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
             --auxdata --w1_dec 128 --end_of_over_v2 --write_rx rx_v2_eoo_test.f32 \
             --append_noise 1 --g_file g_mpp.f32 --EbNodB 8; \
             cat rx_v2_eoo_test.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 2>&1 >/dev/null")
         set_tests_properties(rade_c_v2_rx_eoo PROPERTIES PASS_REGULAR_EXPRESSION "End-of-over detected")

# V2 C receiver SNR estimator: high SNR (EbNodB=12, mirrors v2_snr_est_high)
add_test(NAME rade_c_v2_rx_snr_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/david_vk5dgr.wav snr_feats.f32; \
             python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth snr_feats.f32 /dev/null \
             --latent-dim 56 --w1_dec 128 --auxdata --rate_Fs --cp 0.004 \
             --time_offset -16 --correct_time_offset -16 --peak \
             --EbNodB 12 --write_rx snr_rx_high.f32 --append_noise 1 > snr_inf_high.txt; \
             SNR3K=\$(grep Measured snr_inf_high.txt | awk '{print \$4}'); \
             cat snr_rx_high.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 \
             --write_snr_est snr_est_v2_c_high.f32 >/dev/null; \
             python3 check_snr_est.py \$SNR3K snr_est_v2_c_high.f32 3.0")
         set_tests_properties(rade_c_v2_rx_snr_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 C receiver SNR estimator: low SNR (EbNodB=2, mirrors v2_snr_est_low)
add_test(NAME rade_c_v2_rx_snr_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
             ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/david_vk5dgr.wav snr_feats.f32; \
             python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth snr_feats.f32 /dev/null \
             --latent-dim 56 --w1_dec 128 --auxdata --rate_Fs --cp 0.004 \
             --time_offset -16 --correct_time_offset -16 --peak \
             --EbNodB 2 --write_rx snr_rx_low.f32 --append_noise 1 > snr_inf_low.txt; \
             SNR3K=\$(grep Measured snr_inf_low.txt | awk '{print \$4}'); \
             cat snr_rx_low.f32 | ${RADE_C_BUILD_DIR}/src/radae_rx --v2 -v 0 \
             --write_snr_est snr_est_v2_c_low.f32 >/dev/null; \
             python3 check_snr_est.py \$SNR3K snr_est_v2_c_low.f32 2.0")
         set_tests_properties(rade_c_v2_rx_snr_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")


# V1 text round-trip test: callsign/gridsquare via EOO data channel
add_test(NAME rade_c_v1_text
         COMMAND ${RADE_C_BUILD_DIR}/src/rade_v1_text_test)
         set_tests_properties(rade_c_v1_text PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 text round-trip test: streaming text via BPSK data symbol channel
add_test(NAME rade_c_v2_text
         COMMAND ${RADE_C_BUILD_DIR}/src/rade_v2_text_test)
         set_tests_properties(rade_c_v2_text PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# WAV pipeline sanity check: encode speech WAV → RADE WAV → decoded WAV, confirm sync+SNR
add_test(NAME rade_c_v1_wav
         COMMAND sh -c "${RADE_C_BUILD_DIR}/src/rade_tx_wav ${CMAKE_SOURCE_DIR}/wav/brian_g8sez.wav /tmp/rade_nopy_v1_wav_tx.wav && \
                        ${RADE_C_BUILD_DIR}/src/rade_rx_wav /tmp/rade_nopy_v1_wav_tx.wav /tmp/rade_nopy_v1_wav_rx.wav")
         set_tests_properties(rade_c_v1_wav PROPERTIES PASS_REGULAR_EXPRESSION "SNR: [0-9]")

add_test(NAME rade_c_v2_wav
         COMMAND sh -c "${RADE_C_BUILD_DIR}/src/rade_tx_wav --v2 ${CMAKE_SOURCE_DIR}/wav/brian_g8sez.wav /tmp/rade_nopy_v2_wav_tx.wav && \
                        ${RADE_C_BUILD_DIR}/src/rade_rx_wav --v2 /tmp/rade_nopy_v2_wav_tx.wav /tmp/rade_nopy_v2_wav_rx.wav")
         set_tests_properties(rade_c_v2_wav PROPERTIES PASS_REGULAR_EXPRESSION "SNR: [0-9]")

endif() # RADE_C_BUILD_DIR

# RADE V2 -----------------------------------------------------------------------------------------------

# rx2.sh "4 point" tests
add_test(NAME v2_rx2_awgn_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh")
         set_tests_properties(v2_rx2_awgn_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_rx2_awgn_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.03 ./test/v2_spot.sh --EbNodB 1 --hangover 100")
         set_tests_properties(v2_rx2_awgn_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_rx2_mpp_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --g_file g_mpp.f32")
         set_tests_properties(v2_rx2_mpp_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_rx2_mpp_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.03 ./test/v2_spot.sh --g_file g_mpp.f32 --EbNodB 6 --hangover 100")
         set_tests_properties(v2_rx2_mpp_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
        
# repeat v2_rx2_mpp_low with 20ms (half frame) time shift to repeat last test with opposite frame sync
# to visualise FS metrics:
# octave:377>  figure(4); frame_sync=load_f32('frame_sync.f32',2); plot(frame_sync(:,1)); hold on; plot(frame_sync(:,2)); hold off;
add_test(NAME v2_rx2_oppfs
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.03 ./test/v2_spot.sh --g_file g_mpp.f32 --EbNodB 6 --prepend_noise 0.98")
         set_tests_properties(v2_rx2_oppfs PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Pop tests: inject pitch=-3.0 (well below the -1.4 threshold) into features_in before inference.
# The encoder compresses this to ~-1.7 at the decoder output, which is still below -1.4 and triggers pops.
# pop1: limit_pitch clamp in rx2.py suppresses pops (FAIL if POP seen)
# pop2: --nolimit_pitch disables clamp, pops occur (PASS if POP seen)
# Both tests exercise the full radio chain to test the limit_pitch code path in rx2.py.
add_test(NAME v2_rx2_mpp_low_pop1
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_SOURCE_DIR}/build/src/lpcnet_demo -features wav/david_vk5dgr.wav features_in.f32; \
                       python3 inject_bad_pitch.py features_in.f32 features_in_bad.f32 -3.0 5; \
                       python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth features_in_bad.f32 /dev/null \
                       --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
                       --auxdata --w1_dec 128 --EbNodB 100 --write_rx pop_tx.f32; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync pop_tx.f32 tmp.wav")
         set_tests_properties(v2_rx2_mpp_low_pop1 PROPERTIES FAIL_REGULAR_EXPRESSION "POP")
add_test(NAME v2_rx2_mpp_low_pop2
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_SOURCE_DIR}/build/src/lpcnet_demo -features wav/david_vk5dgr.wav features_in.f32; \
                       python3 inject_bad_pitch.py features_in.f32 features_in_bad.f32 -3.0 5; \
                       python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth features_in_bad.f32 /dev/null \
                       --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
                       --auxdata --w1_dec 128 --EbNodB 100 --write_rx pop_tx.f32; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync pop_tx.f32 tmp.wav \
                       --nolimit_pitch")
         set_tests_properties(v2_rx2_mpp_low_pop2 PROPERTIES PASS_REGULAR_EXPRESSION "POP")
        
# tests to exercise limits of timing est to decoder interface (Jan 2026 timing corner case)
# we allow a little bit of extra loss at limits (delta=0.05), but want to trap a complete breakdown (like frame sync failing)

add_test(NAME v2_timing_low_delta1
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; test/make_g.sh;
                       delta=0.05 ./test/v2_timing.sh --g_file g_mpp_low.f32 --fix_delta_hat 32 --quiet")
         set_tests_properties(v2_timing_low_delta1 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_timing_low_delta2
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.05 ./test/v2_timing.sh --g_file g_mpp_low.f32 --fix_delta_hat 48 --quiet")
         set_tests_properties(v2_timing_low_delta2 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_timing_high_delta1
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.05 ./test/v2_timing.sh --g_file g_mpp_high.f32 --fix_delta_hat 32 --quiet")
         set_tests_properties(v2_timing_high_delta1 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_timing_high_delta2
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.05 ./test/v2_timing.sh --g_file g_mpp_high.f32 --fix_delta_hat 48 --quiet")
         set_tests_properties(v2_timing_high_delta2 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# dealing with sample clock differences --------------------------

# useful visualisation:
#  octave:573> Ry_smooth=load_c64('Ry_smooth.c64',160); figure(2); mesh(abs(Ry_smooth(1:600,:)))
#  octave:574> delta_hat=load_f32('delta_hat.f32',1); figure(1); clf; plot(delta_hat);

# shift delta backwards logic
add_test(NAME v2_rx2_samclk_back
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --pad_samples 70 --write_Ry_smooth Ry_smooth.c64")
         set_tests_properties(v2_rx2_samclk_back PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# shift delta forward logic
add_test(NAME v2_rx2_samclk_fwd
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --pad_samples 90 --write_Ry_smooth Ry_smooth.c64")
         set_tests_properties(v2_rx2_samclk_fwd PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# rx sample clock 25 ppm slower than tx, --pad_samples adjusted for a shift delta forwards
add_test(NAME v2_rx2_samclk_slow
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --sample_clock_offset 7999.8 --write_Ry_smooth Ry_smooth.c64 --write_delta_hat delta_hat.f32 --pad_samples 120")
         set_tests_properties(v2_rx2_samclk_slow PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# rx sample clock 25 ppm faster than tx, --pad_samples adjusted for a shift delta backwards
add_test(NAME v2_rx2_samclk_fast
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --sample_clock_offset 8000.2 --write_Ry_smooth Ry_smooth.c64 --write_delta_hat delta_hat.f32 --pad_samples 35")
         set_tests_properties(v2_rx2_samclk_fast PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# frequency drift of -5 Hz/minute, 3.5dB SNR MPP channel
add_test(NAME v2_rx2_freq_drift
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.02 ./test/v2_spot.sh --g_file g_mpp.f32 --EbNodB 10 --df_dt -0.1")
         set_tests_properties(v2_rx2_freq_drift PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# two RADE V2 signals signal1, signal2 close together, check signal2 acquired OK, timing on signal2
# contrived to be right on edge of symbol boundary to make a little harder
add_test(NAME v2_rx2_signal1_signal2
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --verbose --prepend_signal2 --prepend_noise 0.31 --EbNodB 10 --reset_output_on_resync")
         set_tests_properties(v2_rx2_signal1_signal2 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# spot checks with AGC

add_test(NAME v2_rx2_agc_mpp_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./test/v2_spot.sh --g_file g_mpp.f32 --agc --gain 4")
         set_tests_properties(v2_rx2_agc_mpp_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
add_test(NAME v2_rx2_agc_mpp_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       delta=0.03 ./test/v2_spot.sh --g_file g_mpp.f32 --EbNodB 6 --agc --gain 0.25")
         set_tests_properties(v2_rx2_agc_mpp_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# rx2.sh with complex->real conversions, `ch` tool, and --agc, high SNR, note --gain need to get us in ballpark
add_test(NAME v2_rx2_ch
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --ssb_bpf --write_rx brian_tx.f32; \
         cat brian_tx.f32 | python3 f32toint16.py --real --scale 16383 | ${CMAKE_BINARY_DIR}/src/ch - - | python3 int16tof32.py --zeropad > brian_rx.f32
         ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725_ml_sync brian_rx.f32 /dev/null \
         --agc --quiet --write_gain gain.f32 --gain 1.22E-4; \
         python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_out_rx2.f32 --compare --clip_start 50 --clip_end 50")
         set_tests_properties(v2_rx2_ch PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 EOO test: transmit with --end_of_over_v2 (no noise), check rx2.py detects "EOO detected"
add_test(NAME v2_eoo_awgn_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --end_of_over_v2 --write_rx v2_eoo_tx.f32 --append_noise 1; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_tx.f32 /dev/null \
                       2>&1 | grep 'EOO detected' && echo PASS || echo FAIL")
         set_tests_properties(v2_eoo_awgn_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME v2_eoo_awgn_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --end_of_over_v2 --write_rx v2_eoo_tx.f32 --append_noise 1 --EbNodB 6; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_tx.f32 /dev/null \
                       2>&1 | grep 'EOO detected' && echo PASS || echo FAIL")
         set_tests_properties(v2_eoo_awgn_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME v2_eoo_mpp_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --end_of_over_v2 --write_rx v2_eoo_tx.f32 --append_noise 1 --g_file g_mpp.f32; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_tx.f32 /dev/null \
                       2>&1 | grep 'EOO detected' && echo PASS || echo FAIL")
         set_tests_properties(v2_eoo_mpp_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME v2_eoo_mpp_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --end_of_over_v2 --write_rx v2_eoo_tx.f32 --append_noise 1 --g_file g_mpp.f32 --EbNodB 8; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_tx.f32 /dev/null \
                       2>&1 | grep 'EOO detected' && echo PASS || echo FAIL")
         set_tests_properties(v2_eoo_mpp_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# V2 EOO false-trigger tests: normal transmission (no EOO) using a long file, check no false "EOO detected"
add_test(NAME v2_eoo_no_false_trigger_awgn
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --write_rx v2_eoo_notx.f32 --EbNodB 10; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_notx.f32 /dev/null \
                       2>&1 | \
                       grep 'EOO detected' && echo FAIL || echo PASS")
         set_tests_properties(v2_eoo_no_false_trigger_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME v2_eoo_no_false_trigger_mpp
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null --rate_Fs \
                       --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
                       --write_rx v2_eoo_notx.f32 --g_file g_mpp.f32 --EbNodB 10; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_eoo_notx.f32 /dev/null \
                       2>&1 | \
                       grep 'EOO detected' && echo FAIL || echo PASS")
         set_tests_properties(v2_eoo_no_false_trigger_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# SNR estimator tests: compare snr_est_dB from rx2.py to SNR3k on inference.py Measured: line
add_test(NAME v2_snr_est_high
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_SOURCE_DIR}/build/src/lpcnet_demo -features wav/david_vk5dgr.wav snr_feats.f32; \
                       python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth snr_feats.f32 /dev/null \
                       --latent-dim 56 --w1_dec 128 --auxdata --rate_Fs --cp 0.004 \
                       --time_offset -16 --correct_time_offset -16 --peak \
                       --EbNodB 12 --write_rx snr_rx_high.f32 --append_noise 1 > snr_inf_high.txt; \
                       SNR3K=\$(grep Measured snr_inf_high.txt | awk '{print \$4}'); \
                       python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync \
                       snr_rx_high.f32 /dev/null --quiet \
                       --write_snr_est snr_est_high.f32; \
                       python3 check_snr_est.py \$SNR3K snr_est_high.f32 3.0")
         set_tests_properties(v2_snr_est_high PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME v2_snr_est_low
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_SOURCE_DIR}/build/src/lpcnet_demo -features wav/david_vk5dgr.wav snr_feats.f32; \
                       python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth snr_feats.f32 /dev/null \
                       --latent-dim 56 --w1_dec 128 --auxdata --rate_Fs --cp 0.004 \
                       --time_offset -16 --correct_time_offset -16 --peak \
                       --EbNodB 2 --write_rx snr_rx_low.f32 --append_noise 1 > snr_inf_low.txt; \
                       SNR3K=\$(grep Measured snr_inf_low.txt | awk '{print \$4}'); \
                       python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync \
                       snr_rx_low.f32 /dev/null --quiet \
                       --write_snr_est snr_est_low.f32; \
                       python3 check_snr_est.py \$SNR3K snr_est_low.f32 2.0")
         set_tests_properties(v2_snr_est_low PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# tx2.sh test: encode with tx2.sh, decode with rx2.sh, compare to inference.py reference
add_test(NAME v2_tx2
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./tx2.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav v2_tx2_rx.f32; \
                       python3 inference.py 250725/checkpoints/checkpoint_epoch_200.pth features_in.f32 features_out.f32 \
                       --latent-dim 56 --w1_dec 128 --auxdata --rate_Fs --cp 0.004 \
                       --time_offset -16 --correct_time_offset -8 --peak; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_tx2_rx.f32 /dev/null --quiet; \
                       python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_out_rx2.f32 \
                       --clip_start 100 --clip_end 300 --compare --delta 0.05")
         set_tests_properties(v2_tx2 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# tx2.sh channel test: tx2->ch->rx2 vs inference->ch->rx2, verifies tx2 matches inference under channel impairments
# --No -20 gives approximately 6 dB SNR3k for --scale 16383 peak-normalised RADE V2 signal (RMS~11172)
add_test(NAME v2_tx2_ch
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -8 \
                       --auxdata --w1_dec 128 --ssb_bpf --write_rx v2_tx2ch_inf.f32; \
                       cat v2_tx2ch_inf.f32 | python3 f32toint16.py --real --scale 16383 | ${CMAKE_BINARY_DIR}/src/ch - - --No -20 | python3 int16tof32.py --zeropad > v2_tx2ch_inf_rx.f32; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_tx2ch_inf_rx.f32 /dev/null \
                       --agc --gain 1.22E-4 --quiet; \
                       cp features_out_rx2.f32 features_out_rx2_inf.f32; \
                       ./tx2.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav v2_tx2ch_tx.f32 --ssb_bpf; \
                       cat v2_tx2ch_tx.f32 | python3 f32toint16.py --real --scale 16383 | ${CMAKE_BINARY_DIR}/src/ch - - --No -20 | python3 int16tof32.py --zeropad > v2_tx2ch_rx.f32; \
                       ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync v2_tx2ch_rx.f32 /dev/null \
                       --agc --gain 1.22E-4 --quiet; \
                       python3 loss.py features_in.f32 features_out_rx2_inf.f32 --features_hat2 features_out_rx2.f32 \
                       --compare --clip_start 50 --clip_end 50 --delta 0.1")
         set_tests_properties(v2_tx2_ch PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

