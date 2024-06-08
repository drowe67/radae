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

# Basic test of quite a lot of code at rate Rs - pass condition is simply that it runs
add_test(NAME inference_basic
         COMMAND sh -c "./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB 10")

# A rate Fs test where we substitute digital QPSK symbols and check BER; good test of rate Fs subsystem, pilot and CP insertion
add_test(NAME inference_ber
         COMMAND sh -c "./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots \
                         --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test")
         set_tests_properties(inference_ber PROPERTIES PASS_REGULAR_EXPRESSION "BER: 0.000")


