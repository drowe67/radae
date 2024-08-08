#!/usr/bin/env bash
# ota_test.sh
#
# Stored file Over The Air (OTA) test for Radio Autoencoder:
#   + Given an input speech wave file, constructs a chirp-compressed SSB-radae signal tx.wav
#   + Transmit the tx.wav over a COTS radio, receive to a rx.wav
#   + Prcoess rx.wav to measure the SNR, decodes the radae audio and extract SSB for comparison
#
# Setup (you may not need all of these):
# --------------------------------------
#
# 1. Install python3, sox, octave, octave signal processing toolbox
# 2. Clone and build codec2-dev/master, as we use a few utilities
#    cd codec2
#    mkdir build_linux
#    cd build_linux
#    cmake -DUNITTEST=1 ..
#    make ch mksine tlininterp
# 3. Hamlib cli tools (rigctl), and add user to dialout group, e.g. for "david" user:
#      sudo apt install libhamlib-utils
#      sudo adduser david dialout
#      logout/log back in and check "groups" includes dialout
# 4. Test rigctl (change model number for your radio):
#      echo "m" | rigctl -m 3061 -r /dev/ttyUSB0
# 5. Adjust HF radio Tx drive so ALC is just being tickled, set desired RF power:
#      ./ota_test.sh wav/david.wav -x
#      aplay -f S16_LE --device="plughw:CARD=CODEC,DEV=0" tx.wav
#
# Usage
# -----
#
# 1. File based I/O example:
#    ./ota_test.sh wav/peter.wav -x 
#    ~/codec2-dev/build_linux/src/ch tx.wav - --No -20 | sox -t .s16 -r 8000 -c 1 - rx.wav
#    ./ota_test.sh -r rx.wav
#    aplay rx_ssb.wav rx_radae.wav
#
# 2. Use IC-7200 SSB radio to Tx
#    ./ota_test.sh wav/david.wav -g 9 -d -f 14236
#
# 3. Use HackRF to Tx SSB + radae at 144.5 MHz
#    ./ota_test.sh wav/peter.wav -x -h
#    hackrf_transfer -t tx.iq8 -s 4E6 -f 143.5E6 -R
#    Note tx.iq8 has at +1 MHz offset, so we tune the HackRF 1 MHz low
#  

# TODO: way to adjust /build_linux/src for OSX
CODEC2_DEV=${HOME}/codec2-dev
PATH=${PATH}:${CODEC2_DEV}/build_linux/src:${CODEC2_DEV}/build_linux/misc:${PWD}/build/src

which ch || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

kiwi_url=""
port=8074
freq_kHz="7177"
Nbursts=5
model=3061
gain=6
serialPort="/dev/ttyUSB0"
rxwavefile=0
soundDevice="plughw:CARD=CODEC,DEV=0"
tx_file=0
stationid=""
speechFs=16000
setpoint_rms=6000
setpoint_peak=16384
freq_offset=0
peak=1
hackrf=0

source utils.sh

function print_help {
    echo
    echo "Automated Over The Air (OTA) voice test for Radio Autoencoder"
    echo
    echo "  usage ./ota_test.sh -x InputSpeechWaveFile"
    echo "  usage ./ota_test.sh -r RxWaveFile"
    echo "  or:"
    echo "  usage ./ota_voice_test.sh [options]"
    echo
    echo "    -c dev                    The sound device (in ALSA format on Linux, CoreAudio for macOS)"
    echo "    -d                        Debug mode; trace script execution"
    echo "    -g                        SSB (analog) compressor gain"
    echo "    -i StationIDWaveFile      Prepend this file to identify transmission (should be 8kHz mono)"
    ech  "    -k                        Generate HAckRF output file tx.iq8"
    echo "    -o model                  Select radio model number ('rigctl -l' to list)"
    echo "    -r RxWaveFile             Process supplied rx wave file"
    echo "    -s SerialPort             The serial port (or hostname:port) to control SSB radio,"
    echo "                              default /dev/ttyUSB0"
    echo "    -x InputSpeechWaveFile    Generate tx.wav and exit (no SSB radio Tx)"
    echo "    --rms                     Equalise RMS power of RADAE and SSB (default is equal peak power)"
    echo
    exit
}


function run_rigctl {
    command=$1
    model=$2
    echo $command | rigctl -m $model -r $serialPort > /dev/null
    if [ $? -ne 0 ]; then
        echo "Can't talk to Tx"
        clean_up
        exit 1
    fi
}

function clean_up {
    echo "killing KiwiSDR process"
    kill ${kiwi_pid}
    wait ${kiwi_pid} 2>/dev/null
    exit 1
}

function process_rx {
    echo "Process receiver sample"
    # Place results in same path, same file name as inpout file
    filename="${1%.*}"
     
    rx=$(mktemp).wav
    sox $1 -c 1 -r 8k $rx
    # generate spectrogram
    DISPLAY=""; echo "pkg load signal; warning('off', 'all'); \
          s=load_raw('$rx'); \
          plot_specgram(s, 8000, 200, 3000); print('${filename}_spec.jpg', '-djpg'); \
          quit" | octave-cli -p ${CODEC2_PATH}/octave -qf > /dev/null

    # extract chirp at start and estimate C/No
    rx_chirp=$(mktemp)
    sox $rx -t .s16 ${rx_chirp}.raw trim 0 4
    cat  ${rx_chirp}.raw | python3 int16tof32.py --zeropad > ${rx_chirp}.f32
    python3 est_CNo.py ${rx_chirp}.f32

    # 4 sec chirp - 1 sec silence - x sec SSB - 1 sec silence - x sec RADAE
    # start_radae = 4+1+x
    total_duration=$(sox --info -D $rx)
    x=$(python3 -c "x=(${total_duration}-6)/2; print(\"%f\" % x)")
    start_radae=$(python3 -c "start_radae=5+${x}; print(\"%f\" % start_radae)")
    rx_radae=$(mktemp)
    sox $rx ${filename}_ssb.wav trim 5 $x
    sox $rx -t .s16 ${rx_radae}.raw trim $start_radae

    # Use streaming RADAE Rx
    cat ${rx_radae}.raw | python3 int16tof32.py --zeropad > ${rx_radae}.f32
    cat ${rx_radae}.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 2 > features_rx_out.f32
    lpcnet_demo -fargan-synthesis features_rx_out.f32 - | sox -t .s16 -r 16000 -c 1 - ${filename}_radae.wav
}


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -d)
        set -x	
        shift
    ;;
    -f)
        freq_kHz="$2"	
        shift
        shift
    ;;
    --freq_offset)
        freq_offset="$2"	
        shift
        shift
    ;;
    -g)
        gain="$2"	
        shift
        shift
    ;;
    -i)
        stationid="$2"	
        shift
        shift
    ;;
    -k)
        hackrf=1
        shift
    ;;
    -o)
        model="$2"	
        shift
        shift
    ;;
    -p)
        port="$2"	
        shift
        shift
    ;;
    --rms)
        peak=0
        shift
    ;;
    -r)
        rxwavefile=1	
        shift
    ;;
    -x)
        tx_file=1	
        shift
    ;;
    -c)
        soundDevice="$2"
        shift
        shift
    ;;
    -s)
        serialPort="$2"
        shift
        shift
    ;;
    -h)
        print_help	
    ;;
    *)
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

speechFs=16000

if [ $rxwavefile -eq 1 ]; then
    process_rx $1 $freq_offset
    exit 0
fi

speechfile="$1"
if [ ! -f $speechfile ]; then
    echo "Can't find input speech wave file: ${speechfile}!"
    exit 1
fi

# create Tx file ------------------------

# create 400-2000 Hz chirp header used for C/No est.  We generate 4.5s of chirp, to allow for trimming of
# rx wave file - we need >=4 seconds of received chirp for C/No est at Rx
tx_chirp=$(mktemp)
if [ $peak -eq 1 ]; then
    amp=$(python3 -c "import numpy as np; amp=0.25*${setpoint_peak}/8192.0; print(\"%f\" % amp)")
else
    # TODO: rms option untested as of June 2024
    amp=$(python3 -c "import numpy as np; amp=0.25*${setpoint_rms}*sqrt(2.0)/8192.0; print(\"%f\" % amp)")
fi
python3 chirp.py ${chirp}.f32 4.5 --amp ${amp}
cat ${chirp}.f32 | python3 f32toint16.py --real > ${chirp}.raw

# create compressed SSB signal
speechfile_raw_8k=$(mktemp)
comp_in=$(mktemp)
tx_ssb=$(mktemp)
tx_radae=$(mktemp)
# With 16kHz input files, we need an 8kHz version for SSB
sox $speechfile -r 8000 -t .s16 -c 1 $speechfile_raw_8k
if [ -z $stationid ]; then
    cp $speechfile_raw_8k $comp_in
else
    # append station ID
    stationid_raw_8k=$(mktemp)
    sox $stationid -r 8000 -t .s16 -c 1 $stationid_raw_8k
    cat  $stationid_raw_8k $speechfile_raw_8k > $comp_in
fi
analog_compressor $comp_in $tx_ssb $gain

# insert an extra second of silence at start of radae speech input to make sync easier
speechfile_pad=$(mktemp).wav
sox $speechfile $speechfile_pad pad 1@0

# create modulated radae signal
./inference.sh model17/checkpoints/checkpoint_epoch_100.pth $speechfile_pad /dev/null --EbNodB 100 --bottleneck 3 --pilots --cp 0.004 --rate_Fs --write_rx ${tx_radae}.f32
# extract real (I) channel
cat ${tx_radae}.f32 | python3 f32toint16.py --real --scale 16383 > ${tx_radae}.raw 

# Make power of both signals the same, by adjusting the levels to meet the setpoint
if [ $peak -eq 1 ]; then
  set_peak $tx_ssb $setpoint_peak
  set_peak ${tx_radae}.raw $setpoint_peak
else
  set_rms $tx_ssb $setpoint_rms
  set_rms ${tx_radae}.raw $setpoint_rms
fi

# insert 1 second of silence between signals
sox -t .s16 -r 8k -c 1 $tx_ssb -t .s16 -r 8k -c 1 ${tx_ssb}_pad.raw pad 1@0
sox -t .s16 -r 8k -c 1 ${tx_radae}.raw -t .s16 -r 8k -c 1 ${tx_radae}_pad.raw pad 1@0

# cat signals together so we can send them over a radio at the same time
cat ${chirp}.raw ${tx_ssb}_pad.raw ${tx_radae}_pad.raw > tx.raw
sox -t .s16 -r 8000 -c 1 tx.raw tx.wav

# generate a 4MSP .iq8 file suitable for replaying by HackRF (can disable if not using HackRF)
if [ $hackrf -eq 1 ]; then
  ch tx.raw - --complexout | tsrc - - 5 -c | tlininterp - tx.iq8 100 -d -f
fi

if [ $tx_file -eq 1 ]; then
  exit 0
fi

# transmit using local SSB radio

echo "Tx data signal"
freq_Hz=$((freq_kHz*1000))
usb_lsb_upper=$(echo ${usb_lsb} | awk '{print toupper($0)}')
run_rigctl "\\set_freq ${freq_Hz}" $model
run_rigctl "\\set_ptt 1" $model
if [ `uname` == "Darwin" ]; then
    AUDIODEV="${soundDevice}" play -t raw -b 16 -c 1 -r 8000 -e signed-integer --endian little tx.raw 
else
    aplay --device="${soundDevice}" -f S16_LE tx.raw 2>/dev/null
fi
if [ $? -ne 0 ]; then
    run_rigctl "\\set_ptt 0" $model
    clean_up
    echo "Problem running aplay!"
    echo "Is ${soundDevice} configured as the default sound device in Settings-Sound?"
    exit 1
fi
run_rigctl "\\set_ptt 0" $model

