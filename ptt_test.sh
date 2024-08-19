#!/bin/bash

echo "---------------------"
echo "Available audio cards"
echo "---------------------\n"

aplay -l | grep card

printf "\nselect mic card number: "
read -N 1 mic_card_num
printf "\nselect rig input card number: "
read -N 1 rig_card_num
printf "\n"

mic_card=$(aplay -l | grep "card ${mic_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)
rig_card=$(aplay -l | grep "card ${rig_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)

echo $mic_card
echo $rig_card

arecord --device "plughw:CARD=${mic_card},DEV=0" -f S16_LE -c 1 -r 16000 | \
./build/src/lpcnet_demo -features - - | \
python3 radae_tx.py model19_check3/checkpoints/checkpoint_epoch_100.pth --auxdata | \
python3 f32toint16.py --real --scale 8192 | aplay -f S16_LE --device "plughw:CARD=${rig_card},DEV=0"

echo "RADAE transmitter running!  Ctrl-C to exit"
