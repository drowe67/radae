#!/bin/bash -x
#
# RADE V2 wrong carrier detector, all C signal chain with a real valued (SSB radio like) channel:
# C transmitter (radae_tx --v2) -> real int16 -> ch (freq offset, optional multipath, noise) ->
# C receiver (radae_rx --v2, default AGC on). Independent of the Python channel simulation.
#
# usage: ./test/v2_c_wrong_k_ch.sh rade_c_build_src freq_offset_Hz expect [ch options]
#   expect: oor   out of range: PASS if no sync (n_acq 0) and mistune has the sign of freq_offset
#           in    in range: PASS if sync (n_acq >= 1), with no vetoes or in sync drops
#   e.g. ./test/v2_c_wrong_k_ch.sh ~/rade_c/build/src -80 oor --mpp --fading_dir .
#
# scale 8192 + No -30 gives ~10 dB SNR3k with no clipping (see rade_c_v2_rx_eoo_mpp)

RADE_C=$1
f_off=$2
expect=$3
shift; shift; shift

build/src/lpcnet_demo -features wav/brian_g8sez.wav features_wk_ch.f32
cat features_wk_ch.f32 | ${RADE_C}/radae_tx --v2 > tx_wk_ch.f32
cat tx_wk_ch.f32 | python3 f32toint16.py --real --scale 8192 > tx_wk_ch.raw
build/src/ch tx_wk_ch.raw rx_wk_ch.raw --No -30 --freq $f_off "$@"
python3 int16tof32.py --zeropad < rx_wk_ch.raw > rx_wk_ch.f32
cat rx_wk_ch.f32 | ${RADE_C}/radae_rx --v2 -v 0 > /dev/null 2> rx_wk_ch.log
cat rx_wk_ch.log | grep "n_acq\|n_veto"

n_acq=$(grep "n_acq:" rx_wk_ch.log | sed 's/n_acq: //')
n_veto=$(grep "n_veto:" rx_wk_ch.log | sed 's/.*n_veto: \([0-9]*\).*/\1/')
n_wk_drop=$(grep "n_wk_drop:" rx_wk_ch.log | sed 's/.*n_wk_drop: \([0-9]*\).*/\1/')
mistune=$(grep "mistune:" rx_wk_ch.log | sed 's/.*mistune: \(-*[0-9]*\).*/\1/')
f_sign=$(python3 -c "print(1 if $f_off > 0 else -1)")

pass=0
case $expect in
  oor)  [ "$n_acq" -eq 0 ] && [ "$mistune" -eq "$f_sign" ] && pass=1 ;;
  in)   [ "$n_acq" -ge 1 ] && [ "$n_veto" -eq 0 ] && [ "$n_wk_drop" -eq 0 ] && pass=1 ;;
esac
if [ $pass -eq 1 ]; then echo "PASS"; else echo "FAIL"; fi
