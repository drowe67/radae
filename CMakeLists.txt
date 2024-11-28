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

if(NOT CODEC2_DEV_BUILD_DIR)
    set(CODEC2_DEV_BUILD_DIR $HOME/codec2-dev/build_linux )
endif()

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

if(NOT CMAKE_CROSSCOMPILING)
    # Python tells us the CFLAGS we need for Embedding Python in a C lib.
    # We also need to make sure we select the libraries in any activated venv first,
    # in case the user is using that for e.g. NumPy/PyTorch.
    set(Python3_FIND_STRATEGY LOCATION)
    set(Python3_FIND_REGISTRY LAST)
    set(Python3_FIND_FRAMEWORK LAST)
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
else(NOT CMAKE_CROSSCOMPILING)
    # Cross-compiling. find_package(Python3) will find the host's Python,
    # not the target's, so we can't use it. The user will need to specify
    # the root directory of a target's Python.
    if (NOT Python3_ROOT_DIR)
        message(FATAL_ERROR "Must provide -DPython3_ROOT_DIR when cross-compiling.")
    endif (NOT Python3_ROOT_DIR)
    
    add_library(Python3::Python SHARED IMPORTED)
    add_library(Python3::NumPy STATIC IMPORTED)
    if(WIN32)
        target_include_directories(Python3::Python INTERFACE ${Python3_ROOT_DIR}/include)
        target_include_directories(Python3::NumPy INTERFACE ${Python3_ROOT_DIR}/Lib/site-packages/numpy/_core/include)
        set_property(TARGET Python3::Python PROPERTY IMPORTED_IMPLIB ${Python3_ROOT_DIR}/libs/python312.lib)
        set_property(TARGET Python3::Python PROPERTY IMPORTED_LOCATION ${Python3_ROOT_DIR}/python312.dll)
        set_property(TARGET Python3::NumPy PROPERTY IMPORTED_IMPLIB ${Python3_ROOT_DIR}/Lib/site-packages/numpy/core/lib/npymath.lib)
        set_property(TARGET Python3::NumPy PROPERTY IMPORTED_LOCATION ${Python3_ROOT_DIR}/Lib/site-packages/numpy/core/lib/npymath.lib)
    else(WIN32)
        message(FATAL_ERROR "Currently only supports Windows when cross-compiling.")
    endif(WIN32)
endif(NOT CMAKE_CROSSCOMPILING)

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
                       ./test/chirp_mpp.sh ${CODEC2_DEV_BUILD_DIR} -16")

# Low SNR ota_test.sh, with chirp measurement, AWGN
add_test(NAME ota_test_awgn
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ${CODEC2_DEV_BUILD_DIR} -21 0.4")

# Low SNR ota_test.sh, with chirp measurement, MPP, high loss threshold as we only care about gross errors,
# like stuck in false sync
add_test(NAME ota_test_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./test/ota_test_cal.sh ${CODEC2_DEV_BUILD_DIR} -24 0.4 --mpp --freq -25")


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
# really just trying to exercise the slip code here, not too cocnerned about loss.  There should be a noticable transition in tmax in each test as
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

# Test ability to handle buffer slips due to sample clock offsets, rx ADC clock < tx ADC clock
add_test(NAME radae_rx_slip_minus
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 10 --freq_offset 31 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset --prepend_noise 0.11; \
                       cat rx.f32 | python3 f32toint16.py --scale 8192 | sox -t .s16 -r 8000 -c 2 - -t .s16 -r 7995 -c 2 - | python3 int16tof32.py > rx_.f32; \
                       cat rx_.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rxs_out.f32; \
                       python3 loss.py features_in.f32 features_rxs_out.f32 --loss_test 0.2 --acq_time_test 1.0")
                       set_tests_properties(radae_rx_slip_minus PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# profiles a run with a 50 second file (no pass/fail, run with -V to get a rough idea of execution time)
add_test(NAME radae_rx_profile
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --EbNodB 1 --freq_offset 13 --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over; \
                       cat rx.f32 | python3 -m cProfile -s time radae_rxe.py -v 1 --no_stdout | head -n20")

# Characterise run time using full simplex Tx stack (C core encoder version).  No pass/fail, just for characterisation of run time
add_test(NAME radae_tx_stack_c
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       \\time -o log.txt -f '%e' ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features $WAV - | 
                       PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_tx > /dev/null; \
                       RUN_TIME=$(cat log.txt); DUR=$(soxi -D $WAV); percent=$(python3 -c \"percent=100*$RUN_TIME/$DUR; print('%f' % percent)\"); \
                       printf \"\nrun time: %5.2f duration: %5.2f percent CPU: %5.2f\n\n\" $RUN_TIME $DUR $percent ")

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

# As above but with C core decoder for comparison
add_test(NAME radae_rx_stack_c
        COMMAND bash -c "WAV='wav/all.wav'; cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth $WAV /dev/null \
                       --EbNodB 10 --freq_offset 13 --df_dt -0.1 --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --write_rx rx.f32  \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset ; \
                       \\time -o log.txt -f '%e' cat rx.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_rx | \
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


# Embedding Python in C callable library ------------------------------------------------------------------------------------------------------

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

# Test Embedded version of streaming Tx, running with C main()
add_test(NAME radae_tx_embed_c
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx rx.f32 --correct_freq_offset; \
                       cat features_in.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_tx > rx.f32; \
                       cat rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 1 > features_txs_out.f32; \
                       python3 loss.py features_in.f32 features_txs_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_tx_embed_c PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Test Embedded version of streaming Rx, running with C main()
add_test(NAME radae_rx_embed_c
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 100 --freq_offset 0 --append_noise 1 --end_of_over --auxdata \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32 --correct_freq_offset; \
                       cat rx.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_rx > features_out.f32;
                       python3 loss.py features_in.f32 features_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(radae_rx_embed_c PROPERTIES PASS_REGULAR_EXPRESSION "PASS")


# C Port of Core Encoder/decoder ------------------------------------------------------------------------------------

if (NOT WIN32)
# we test by comparing loss of features_in/features_out, can all happen at rate Rs.  We load model05.bin weights as 
# compiled-in weights are for model19_check3
add_test(NAME c_encoder_model5
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
         ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
         cat features_in.f32 | ${CMAKE_CURRENT_BINARY_DIR}/src/test_rade_enc 1 0 ${CMAKE_SOURCE_DIR}/bin/model05.bin > z_c.f32; \
         python3 stateful_encoder.py model05/checkpoints/checkpoint_epoch_100.pth features_in.f32 /dev/null --read_latent z_c.f32 --loss_test 0.2")
         set_tests_properties(c_encoder_model5 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# we need the DSP code in the loop to test the core encoder with model19_check3 as the bottleneck (3) is at rate Fs
add_test(NAME c_encoder_model19_check3
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
         ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
         cat features_in.f32 | python3 radae_txe.py | python3 radae_rxe.py > features_out.f32; \
         cat features_in.f32 | ${CMAKE_CURRENT_BINARY_DIR}/src/test_rade_enc 3 1 > z_c.f32; \
         cat z_c.f32 | python3 radae_txe.py --bypass_enc | python3 radae_rxe.py > features_rx_out.f32; \
         python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_rx_out.f32 --compare")
         set_tests_properties(c_encoder_model19_check3 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# compare Python decoder (features_out.f32) C decoder (features_c.f32), for same latent inputs z.f32
# note inference.sh creates features_in.f32 & features_out.f32
add_test(NAME c_decoder_model5
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
         ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --write_latent z.f32; \
         cat z.f32 | ${CMAKE_CURRENT_BINARY_DIR}/src/test_rade_dec 0 ${CMAKE_SOURCE_DIR}/bin/model05.bin > features_c.f32; \
         python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_c.f32 --compare")
         set_tests_properties(c_decoder_model5 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# As well as testing C core decoder, this test also uses radae_rxe.py in bypass mode
add_test(NAME c_decoder_model19_check3
         COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; 
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --correct_freq_offset \
                       --write_rx rx.f32; \
         cat rx.f32 | python3 radae_rxe.py --bypass_dec > z_hat.f32
         cat z_hat.f32 | ${CMAKE_CURRENT_BINARY_DIR}/src/test_rade_dec 1 > features_c.f32; \
         python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_c.f32 --compare")
         set_tests_properties(c_decoder_model19_check3 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
endif(NOT WIN32)

# Test embedded data (--auxdata) use for false sync detection, with C core decoder, with C callable embedded Python API.
# --foff_err forces a false sync state after first sync. See also "radae_rx_aux_mpp" above.  Tests uw_errors are being passed
# back from C decoder to Python state machine
add_test(NAME c_decoder_aux_mpp
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       test/make_g.sh; \
                       ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --auxdata \
                       --EbNodB 4 --freq_offset -11 --g_file g_mpp.f32 --write_rx rx.f32 \
                       --prepend_noise 1 --append_noise 3 --end_of_over --correct_freq_offset; \
                       cat rx.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_rx 1 > features_c.f32; \
                       python3 loss.py features_in.f32 features_c.f32 --loss 0.3 --clip_start 300")
                       set_tests_properties(c_decoder_aux_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# EOO data -------------------------------------------------------------------------------------------

add_test(NAME radae_eoo_data_py
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
                       cat features_in.f32 | python3 radae_txe.py --eoo_data_test > rx.f32; \
                       cat rx.f32 | python3 radae_rxe.py -v 2 --eoo_data_test > /dev/null")
                       set_tests_properties(radae_eoo_data_py PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# C tx. C rx. Note Python radae_txe.py just generates eoo_tx.f32 for C radae_tx
add_test(NAME radae_eoo_data_c
        COMMAND sh -c "cd ${CMAKE_SOURCE_DIR}; \
                       ${CMAKE_CURRENT_BINARY_DIR}/src/lpcnet_demo -features wav/brian_g8sez.wav features_in.f32; \
                       cat features_in.f32 | python3 radae_txe.py --eoo_data_test > /dev/null; \
                       cat features_in.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_tx > rx.f32; \
                       cat rx.f32 | PYTHONPATH='.' ${CMAKE_CURRENT_BINARY_DIR}/src/radae_rx > /dev/null; \
                       python3 eoo_ber.py eoo_tx.f32 eoo_rx.f32")
                       set_tests_properties(radae_eoo_data_c PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

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
                       cat z.f32 | python3 sc_tx.py | ${CODEC2_DEV_BUILD_DIR}/src/ch - - | python3 sc_rx.py > z_hat.f32; \
                       ./bbfm_rx.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 /dev/null; \
                       python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_rx_out.f32 --compare")
                       set_tests_properties(bbfm_sc_bpf_loss PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

