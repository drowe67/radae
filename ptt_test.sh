#!/bin/bash
#
# Script based console RADAE application for sending RADAE over a SSB radio
#
# Scope: This is not production software.  It is designed for the Linux user who is comfortable 
# experimenting/modifying this script to get it running on their station.  The goal is just enough
# functionality to try real time RADAE contacts over the air.
#
# Notes:
#   1. Set up ptt_test.conf
#   2. t to start/stop transmitting
#   3. r to start/stop receiving
#   4. When idle, press S to enter SSB mode (audio piped through).
#   5. Note SSB Tx won't be compressed unless you use the radios compressor
#   6. Note radios Tx compressor should be off RADAE Tx (clean audio path)
#   7. To listen from a KiwSDR
#      sudo modprobe snd-aloop
#      Open your Settings, and select the loopback device this was obscurely labelled "Analog Output - Built in Audio" on my machine.
#      In ptt_test.conf use the Loopback dev# for rig_out_card_num, and rig_out_dev 1 

# Hamlib rigctl model and serial port, change by setting an env variable
conf=${conf:-"ptt_test.conf"}
model=$(cat $conf | grep "rigctl_model" | tr -s ' ' | cut -d' ' -f2)
serial=$(cat $conf | grep "serial_port" | tr -s ' ' | cut -d' ' -f2)

function clean_up {
    run_rigctl "\\set_ptt 0" $model
    echo "killing Tx process"
    kill `cat tx_pid`
    wait ${tx_pid} 2>/dev/null
    echo "killing Rx process"
    kill `cat rx_pid`
    wait ${tx_pid} 2>/dev/null
    exit 1
}

function kick_off_ssb_tx_process {
    ( arecord --device "plughw:CARD=${mic_card},DEV=0" -f S16_LE -c 1 -r 8000 & echo $! >tx_pid ) | \
    aplay -f S16_LE --device "plughw:CARD=${rig_in_card},DEV=0" &

    echo
    echo "Starting SSB transmitter...."
    echo
    cat tx_pid
}

function kick_off_radae_tx_process {
    # note we store PID of process before radae_tx.py, killing should mean end of over pilot sequence is sent by radae_tx.py (TBC)
    ( arecord --device "plughw:CARD=${mic_card},DEV=0" -f S16_LE -c 1 -r 16000 & echo $! >tx_pid ) | \
    ./build/src/lpcnet_demo -features - - | \
    python3 radae_tx.py model19_check3/checkpoints/checkpoint_epoch_100.pth --auxdata | \
    python3 f32toint16.py --real --scale 8192 | aplay -f S16_LE --device "plughw:CARD=${rig_in_card},DEV=0" &

    echo
    echo "Starting RADAE transmitter...."
    echo
    cat tx_pid
}

function kick_off_radae_rx_process {
    ( arecord --device "plughw:CARD=${rig_out_card},DEV=${rig_out_card_dev}" -f S16_LE -c 1 -r 8000 & echo $! >rx_pid ) | \
    python3 int16tof32.py --zeropad | \
    python3 radae_rx.py model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --auxdata | \
    ./build/src/lpcnet_demo -fargan-synthesis - - | \
    aplay -f S16_LE -r 16000 --device "plughw:CARD=${speaker_card},DEV=0" &

    echo
    echo "Starting RADAE receiver...."
    echo
    cat rx_pid
}

function kick_off_ssb_rx_process {
    ( arecord --device "plughw:CARD=${rig_out_card},DEV=${rig_out_card_dev}" -f S16_LE -c 1 -r 8000 & echo $! >rx_pid ) | \
    aplay -f S16_LE -r 8000 --device "plughw:CARD=${speaker_card},DEV=0" &

    echo
    echo "Starting SSB receiver...."
    echo
    cat rx_pid
}

function run_rigctl {
    command=$1
    model=$2
    echo $command | rigctl -m $model -r $serial > /dev/null
    if [ $? -ne 0 ]; then
        echo "Can't talk to Tx"
        exit 1
    fi
}

echo "---------------------------------"
echo "Available audio cards and devices"
echo "---------------------------------"
echo 

aplay -l | grep card

echo 
echo "---------------------------------"
echo "Loading ${conf}"
echo "---------------------------------"
echo 

mic_card_num=$(cat $conf | grep "mic_card_num" | tr -s ' ' | cut -d' ' -f2)
mic_card_dev=$(cat $conf | grep "mic_card_dev" | tr -s ' ' | cut -d' ' -f2)
rig_in_card_num=$(cat $conf | grep "rig_in_card_num" | tr -s ' ' | cut -d' ' -f2)
rig_in_card_dev=$(cat $conf | grep "rig_in_card_dev" | tr -s ' ' | cut -d' ' -f2)
rig_out_card_num=$(cat $conf | grep "rig_out_card_num" | tr -s ' ' | cut -d' ' -f2)
rig_out_card_dev=$(cat $conf | grep "rig_out_card_dev" | tr -s ' ' | cut -d' ' -f2)
speaker_card_num=$(cat $conf | grep "speaker_card_num" | tr -s ' ' | cut -d' ' -f2)
speaker_card_dev=$(cat $conf | grep "speaker_card_dev" | tr -s ' ' | cut -d' ' -f2)

mic_card=$(aplay -l | grep "card ${mic_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)
rig_in_card=$(aplay -l | grep "card ${rig_in_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)
rig_out_card=$(aplay -l | grep "card ${rig_out_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)
speaker_card=$(aplay -l | grep "card ${speaker_card_num}" | head -n 1 | tr -s ' ' | cut -d' ' -f3)

printf "rigctl_model: %d\n" $model
printf "serial_port.: %s\n" $serial

printf "mic....: %d: %-10s device %d\n" $mic_card_num $mic_card $mic_card_dev
printf "rig_in.: %d: %-10s device %d\n" $rig_in_card_num  $rig_in_card $rig_in_card_dev
printf "rig_out: %d: %-10s device %d\n" $rig_out_card_num $rig_out_card $rig_out_card_dev
printf "speaker: %d: %-10s device %d\n" $speaker_card_num $speaker_card $speaker_card_dev

# clean up any processes if we get a ctrl-C
trap clean_up SIGHUP SIGINT SIGTERM

rx=0
tx=0
ssb=0
key=' ' 
while [[ ! $key == "q" ]]
do
    printf "\n***************************************************************\n"
    printf "tx: %d rx: %d ssb: %d Quit-q Toggle Tx-t Toggle Rx-r Toggle SSB-s\n" $tx $rx $ssb
    printf "***************************************************************\n"
    read -N 1 key

    # toggle SSB/RADAE
    if [ "$key" == "s" ]; then
        echo "toggle SSB"
        if [ "$ssb" -eq 1 ]; then
            ssb=0
        else
            ssb=1
        fi
    fi

    # toggle transmit 
    if [ "$key" == "t" ]; then
        if [ $tx -eq 1 ]; then
            tx=0
            kill `cat tx_pid`
            echo
            echo 'Tx stopped!'
            echo
            run_rigctl "\\set_ptt 0" $model
        else
            if [ $rx -eq 1 ]; then
                kill `cat rx_pid`
            fi
            tx=1
            if [ $ssb -eq 0 ]; then
                kick_off_radae_tx_process
            else
                kick_off_ssb_tx_process
            fi
            run_rigctl "\\set_ptt 1" $model
        fi
    fi

    # toggle receive
    if [ "$key" == "r" ]; then
        if [ $rx -eq 1 ]; then
            rx=0
            kill `cat rx_pid`
            echo
            echo 'Rx stopped!'
            echo
         else
            if [ $tx -eq 1 ]; then
                kill `cat tx_pid`
                run_rigctl "\\set_ptt 0" $model
            fi
            rx=1
            if [ $ssb -eq 0 ]; then
                kick_off_radae_rx_process
            else
                kick_off_ssb_rx_process
            fi
        fi
    fi
done

if [ $tx -eq 1 ]; then
  kill `cat tx_pid`
fi
if [ $rx -eq 1 ]; then
  kill `cat rx_pid`
fi

