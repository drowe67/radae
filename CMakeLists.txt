cmake_minimum_required(VERSION 3.13)
project(RADAE
  VERSION 0.1
  DESCRIPTION "Radio Autoencoder - transmission of vocoder features over radio channels"
  HOMEPAGE_URL "https://github.com/drowe67/radae"
  LANGUAGES C
  )

if(NOT CODEC2_DEV_BUILD_DIR)
    set(CODEC2_DEV_BUILD_DIR $HOME/codec2-dev/build_linux )
endif()


# Ctests ----------------------------------------------------------------------

include(CTest)
enable_testing()

# Basic test of radae/radae.py code at rate Rs - pass condition is simply that it runs
add_test(NAME inference_model5
         COMMAND sh -c "./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB 10")

# Vanilla tests of model 17 & 18, just to confirm they're working
add_test(NAME inference_model17
         COMMAND sh -c "./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                        --EbNodB 0 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3")
add_test(NAME inference_model18
         COMMAND sh -c "./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --latent-dim 40 \
                        --EbNodB 0 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3")

# Stateful decoder
add_test(NAME stateful_decoder
         COMMAND sh -c "./stateful_decoder.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --loss_test 0.2")
         set_tests_properties(stateful_decoder PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# DIGITAL SYMBOL BER TESTS ----------------------------------------------------------

# Substitute digital QPSK symbols and check BER; tests rate Fs subsystem, pilot and CP insertion, eq_ls

# Vanilla high SNR test
add_test(NAME inference_ber
         COMMAND sh -c "./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots \
                         --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test")
         set_tests_properties(inference_ber PROPERTIES PASS_REGULAR_EXPRESSION "BER: 0.000")

# As above but on AWGN at operating point Eb/No - gives LS EQ a work out
add_test(NAME inference_ber_awgn
         COMMAND sh -c "test/inference_ber_awgn.sh")
         set_tests_properties(inference_ber_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# As above but on MPP at operating point Eb/No
add_test(NAME inference_ber_mpp
         COMMAND sh -c "test/inference_ber_mpp.sh")
         set_tests_properties(inference_ber_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# ML SYMBOL OP POINT LOSS TESTS ----------------------------------------------------------

# all adjusted for -2.5dB-ish SNR operating point AWGN, should be loss < 0.3

add_test(NAME inference_loss_model5
        COMMAND sh -c "./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB -2.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model5 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME inference_loss_model17
        COMMAND sh -c "./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --EbNodB 0.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model17 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

add_test(NAME inference_loss_model18
        COMMAND sh -c "./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --latent-dim 40 \
                       --EbNodB 3.5  --freq_offset 1 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --loss_test 0.3")
                       set_tests_properties(inference_loss_model18 PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# rx.sh/ota_test.sh tests ------------------------------------------------------------------------------------

# Generate rx.f32 rate Fs IQ samples, run through stand alone rx/py receiver, measure loss and acquisition time
add_test(NAME rx_loss_acq_time
        COMMAND sh -c "./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --EbNodB 3  --freq_offset 10 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32; \
                       rm -f features_rx_out.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss_test 0.3 --acq_time_test 0.8")
                       set_tests_properties(rx_loss_acq_time PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Estimating C/No on multipath channels using a chirp
add_test(NAME chirp_mpp
        COMMAND sh -c "./test/chirp_mpp.sh ${CODEC2_DEV_BUILD_DIR} -16")
                       set_tests_properties(chirp_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Low SNR ota_test.sh, with chirp measurement, AWGN
add_test(NAME ota_test_awgn
        COMMAND sh -c "./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -21")
                       set_tests_properties(ota_test_awgn PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
                       set_tests_properties(ota_test_awgn PROPERTIES PASS_REGULAR_EXPRESSION "CNodB PASS")


# Low SNR ota_test.sh, with chirp measurement, MPP
add_test(NAME ota_test_mpp
        COMMAND sh -c "./test/ota_test_cal.sh ~/codec2-dev/build_linux/ -25 --mpp --freq -25")
                       set_tests_properties(ota_test_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
                       set_tests_properties(ota_test_mpp PROPERTIES PASS_REGULAR_EXPRESSION "CNodB PASS")


# Acquisition tests ------------------------------------------------------------------------------------

# noise-only test, should not acquire for 120s (currently set at 10s as it's too slow)
add_test(NAME acq_noise
        COMMAND sh -c "dd if=/dev/zero of=/dev/stdout bs=16000 count=10 | \
                       ${CODEC2_DEV_BUILD_DIR}/src/ch - rx.int16 --No -20; \  # real int16 output
                       python3 int16tof32.py rx.int16 rx.f32 --zeropad; \     # ..IQIQI.. .f32 with Q == 0
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null \
                       --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16")
                       set_tests_properties(acq_noise PROPERTIES PASS_REGULAR_EXPRESSION "Acquisition failed")

# Worst case: 0dB SNR MPP, make sure mean acq time < 1s
add_test(NAME acq_mpp
        COMMAND sh -c "test/make_g_mpp.sh; \
                       ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 \
                       --EbNodB 3 --freq_offset 10 --g_file g_mpp.f32 --write_rx rx.f32; \
                       ./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx.f32 /dev/null --pilots --pilot_eq --bottleneck 3 --cp 0.004 --acq_test --fmax_target 10")
                       set_tests_properties(acq_mpp PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# TODO: Add (a) interfering sine wave (b) stationary notch or perhaps slow fading

# Streaming receiver --------------------------------------------------------------------------------------

# basic test of streaming rx, fairly high SNR
add_test(NAME streaming_rx
        COMMAND sh -c "./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB 10 --freq_offset 11 \
                       --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32; \
                       cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth features_rx_out.f32; \
                       python3 loss.py features_in.f32 features_rx_out.f32 --loss_test 0.15 --acq_time_test 0.5")
                       set_tests_properties(streaming_rx PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# evaluate.sh -----------------------------------------------------------------------------------------------

# TODO test evaluate.sh to prevent bit rot
