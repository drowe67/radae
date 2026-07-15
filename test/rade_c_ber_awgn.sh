#!/bin/bash -x
#
# BER test for radae_nopy C port OFDM modem: AWGN channel

NOPY_BUILD=${1:-$HOME/radae_nopy/build}

EbNodB=0
ILdB=2
ber=$($NOPY_BUILD/src/rade_ber_test --EbNodB $EbNodB --frames 200 | grep BER | tr -s ' ' | cut -d' ' -f2)
echo "EbNo=10^((${EbNodB}-${ILdB})/10); \
      target_ber = 0.5*erfc(sqrt(EbNo)); \
      printf('target: %f measured: %f\n', target_ber, ${ber}); \
      if ${ber} < target_ber printf('PASS\n'); end; quit" | octave-cli -qf
