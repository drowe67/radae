#!/bin/bash

function clean_up {
    echo "killing Tx process"
    kill `cat tx_pid`
    wait ${tx_pid} 2>/dev/null
    exit 1
}

function kick_off_tx_process {
    arecord --device "plughw:CARD=${mic_card},DEV=0" -f S16_LE -c 1 -r 16000 | \
    ./build/src/lpcnet_demo -features - - | \
    ( python3 radae_tx.py model19_check3/checkpoints/checkpoint_epoch_100.pth --auxdata & echo $! >tx_pid ) | \
    python3 f32toint16.py --real --scale 8192 | aplay -f S16_LE --device "plughw:CARD=${rig_card},DEV=0" &

    echo
    echo "RADAE transmitter running!  Any key to exit"
    echo
    cat tx_pid
}

# Set sounds cards every time we start as they may change

echo "---------------------"
echo "Available audio cards"
echo "---------------------"
echo 

aplay -l | grep card

printf "\nselect mic card number......: "
read -N 1 mic_card_num
printf "\nselect rig input card number: "
read -N 1 rig_card_num
printf "\n"

mic_card=$(aplay -l | grep "card ${mic_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)
rig_card=$(aplay -l | grep "card ${rig_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)

echo $mic_card
echo $rig_card

# clean up any processes if we get a ctrl-C
trap clean_up SIGHUP SIGINT SIGTERM

tx=0
key=' ' 
while [[ ! $key == "q" ]]
do
    read -N 1 key
    echo $key
    key=${key/ /t}
    if [ "$key" == "t" ]; then
        if [ $tx -eq 1 ]; then
            tx=0
            kill `cat tx_pid`
            echo
            echo 'Tx stopped!'
            echo
        else
            tx=1
            kick_off_tx_process
        fi
    fi
done

if [ $tx -eq 1 ]; then
  kill `cat tx_pid`
fi

