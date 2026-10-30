#!/bin/bash -x
#
# RADE V2 wrong carrier test. A freq offset outside +/-31.25 Hz aliases the receiver's freq
# estimate by one carrier (62.5 Hz). Without the wrong carrier detector the receiver syncs and
# outputs garbled speech ("R2D2"); with it, the receiver stays out of sync and indicates the
# mistune direction.
#
# usage: ./test/v2_wrong_k.sh freq_offset_Hz expect [inference.py channel options]
#   expect: oor   out of range: PASS if no sync (n_acq 0) and mistune has the sign of freq_offset
#           in    in range: PASS if sync (n_acq >= 1), with no vetoes or in sync drops
#           sync  PASS if sync (n_acq >= 1)
#   RX2_OPTS env variable: extra rx2.py options, e.g. RX2_OPTS=--no_wrong_k
#
# examples:
#   ./test/v2_wrong_k.sh 45 oor
#   ./test/v2_wrong_k.sh 62.5 oor --g_file g_mpp.f32 --EbNodB 6
#   ./test/v2_wrong_k.sh 25 sync --g_file g_mpp.f32 --EbNodB 6 --sine_amp 0.2 --sine_freq 1505

OPUS=build/src
PATH=${PATH}:${OPUS}
f_off=$1
expect=$2
shift; shift

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
--peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx rx_wrong_k.f32 \
--prepend_noise 1 --append_noise 2 --freq_offset $f_off "$@"

python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync rx_wrong_k.f32 /dev/null \
--latent-dim 56 --w1_dec 128 --quiet --write_wrong_k wrong_k.f32 --write_state state.int16 $RX2_OPTS 2> rx_wrong_k.log
cat rx_wrong_k.log | grep "n_acq\|n_veto"

n_acq=$(grep "n_acq:" rx_wrong_k.log | sed 's/n_acq: //')
n_veto=$(grep "n_veto:" rx_wrong_k.log | sed 's/.*n_veto: \([0-9]*\).*/\1/')
n_wk_drop=$(grep "n_wk_drop:" rx_wrong_k.log | sed 's/.*n_wk_drop: \([0-9]*\).*/\1/')
mistune=$(grep "mistune:" rx_wrong_k.log | sed 's/.*mistune: \(-*[0-9]*\).*/\1/')
f_sign=$(python3 -c "print(1 if $f_off > 0 else -1)")

pass=0
case $expect in
  oor)  [ "$n_acq" -eq 0 ] && [ "$mistune" -eq "$f_sign" ] && pass=1 ;;
  in)   [ "$n_acq" -ge 1 ] && [ "$n_veto" -eq 0 ] && [ "$n_wk_drop" -eq 0 ] && pass=1 ;;
  sync) [ "$n_acq" -ge 1 ] && pass=1 ;;
esac
if [ $pass -eq 1 ]; then echo "PASS"; else echo "FAIL"; fi
