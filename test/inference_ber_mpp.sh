#!/bin/bash -x
#
# test inference.sh BER at MPP operating point

if [ ! -f g_mpp.f32 ]; then
  DISPLAY="" echo "Fs=8000; Rs=50; Nc=20; multipath_samples('mpp', Fs, Rs, Nc, 120, 'h_nc20_mpp.f32','g_mpp.f32'); quit" | octave-cli -qf
fi

EbNodB=0
ILdB=2
tmp=$(mktemp);
./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --EbNodB ${EbNodB} --freq_offset 1 \
--cp 0.004 --pilots --pilot_eq --eq_ls --ber_test --g_file g_mpp.f32 > $tmp
ber=$(cat ${tmp} | grep 'BER' | tr -s ' ' | cut -d' ' -f4)
echo "EbNo=10^((${EbNodB}-${ILdB})/10); \
      target_ber =0.5*(1 - sqrt(EbNo/(EbNo+1))); \
      printf('target: %f\n', target_ber); \
      if ${ber} < target_ber printf('PASS\n'); end; quit" | octave-cli -qf


