cmake_minimum_required(VERSION 3.13)
project(RADAE
  VERSION 0.1
  DESCRIPTION "Radio Autoencoder - transmission of vocoder features over radio channels"
  HOMEPAGE_URL "https://github.com/drowe67/radae"
  LANGUAGES C
  )

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

# all adjusted for -2.5dB-ish SNR operating point, should be loss < 0.3

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

# TODO test evaluate.sh to prevent bit rot
# TODO test ota_test.sh to prevent bit rot

