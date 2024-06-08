#!/bin/bash -x
#
# test inference.sh BER at AWGN operating point

EbNodB=0
ILdB=2
tmp=$(mktemp);
./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --EbNodB ${EbNodB} --freq_offset 1 \
 --cp 0.004 --pilots --pilot_eq --eq_ls --ber_test > $tmp
ber=$(cat ${tmp} | grep 'BER' | tr -s ' ' | cut -d' ' -f4)
echo "target_ber = 0.5*erfc(sqrt(10^((${EbNodB}-${ILdB})/10))); \
      printf('target: %f\n', target_ber); \
      if ${ber} < target_ber printf('PASS\n'); end; quit" | octave-cli -qf


