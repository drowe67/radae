#!/usr/bin/env bash
# ota_test.sh
#
# Stored file Over The Air (OTA) test for Radio Autoencoder:
#   + Given an input speech wave file, constructs a chirp-compressed SSB-rade1-rade2 signal tx.wav
#   + Transmit the tx.wav over a COTS radio, receive to a rx.wav
#   + Process rx.wav to measure the SNR, decodes the radae audio and extract SSB for comparison
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
# 5. Using Settings to make sure default sound device is not the radio
# 6. Adjust HF radio Tx drive so ALC is just being tickled, set desired RF power:
#      ./ota_test.sh wav/david_vk5dgr.wav -x
#      aplay -f S16_LE --device="plughw:CARD=CODEC,DEV=0" tx.wav
#
# Usage
# -----
#
# 1. File based I/O example:
#    ./ota_test.sh wav/peter.wav -x 
#    ~/codec2-dev/build_linux/src/ch tx.wav - --No -20 | sox -t .s16 -r 8000 -c 1 - rx.wav
#    ./ota_test.sh -r rx.wav
#    aplay rx_ssb.wav rx_radae1.wav rx_radae2.wav
#
# 2. Use IC-7200 SSB radio to Tx
#    ./ota_test.sh wav/david_vk5dgr.wav -d -f 14236
# 
# 3. Process file rx.wav received off. First use a wav file editor to trim any silence from start, then: 
#    ./ota_test.sh -r rx.wav
#    Then listen to rx_ssb.wav and rx_radae.wav
#
# 4. Use HackRF to Tx SSB + radae at 144.5 MHz
#    ./ota_test.sh wav/peter.wav -x -h
#    hackrf_transfer -t tx.iq8 -s 4E6 -f 143.5E6 -R
#    Note tx.iq8 has at +1 MHz offset, so we tune the HackRF 1 MHz low
#  

# TODO: way to adjust /build_linux/src for OSX
CODEC2_DEV=${CODEC2_DEV:-${HOME}/codec2-dev}
PATH=${PATH}:${CODEC2_DEV}/build_linux/src:${CODEC2_DEV}/build_linux/misc:${PWD}/build/src

which ch >/dev/null || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

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
tx_path="."
just_tx=0
loss=""

source utils.sh

function print_help {
    echo
    echo "Automated Over The Air (OTA) voice test for Radio Autoencoder"
    echo
    echo "  usage ./ota_test.sh InputSpeechWaveFile (prepare tx.wav and Tx using radio)"
    echo "  usage ./ota_test.sh -x InputSpeechWaveFile (prepare tx.wav)"
    echo "  usage ./ota_test.sh -r RxWaveFile (process RxWaveFile)"
    echo
    echo "    -c dev                    The sound device (in ALSA format on Linux, CoreAudio for macOS)"
    echo "    -d                        Debug mode; trace script execution"
    echo "    -g                        SSB (analog) compressor gain"
    echo "    -i StationIDWaveFile      Prepend this file to identify transmission (should be 8kHz mono)"
    echo "    -k                        Generate HackRF output file tx.iq8"
    echo "    -l InputSpeechWaveFile    (rx) calculate loss using previously generated feature files" 
    echo "    -o model                  Select radio model number ('rigctl -l' to list)"
    echo "    -r RxWaveFile             Process supplied rx wave file"
    echo "    -s SerialPort             The serial port (or hostname:port) to control SSB radio,"
    echo "                              default /dev/ttyUSB0"
    echo "    -x InputSpeechWaveFile    Generate tx.wav and exit (no SSB radio Tx)"
    echo "    --rms                     Equalise RMS power of RADAE and SSB (default is equal peak power)"
    echo "    --tx_path                 optional path to tx.raw/tx.wav"
    echo "    -t SSBRadioFile.raw       Tx SSBRadioFile.raw over SSB radio (e.g. tx.wav or RADAE encoded file), no pre-processing"
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
    run_rigctl "\\set_ptt 0" $model
}

function process_rx {
    echo "-----------------------------------------------"
    echo "Process receiver sample"
    echo "-----------------------------------------------"
    # Place results in same path, same file name as input file
    filename="${1%.*}"
     
    rx=$(mktemp).wav
    sox $1 -c 1 -r 8k $rx
    # generate spectrogram
    DISPLAY=""; echo "pkg load signal; warning('off', 'all'); \
          s=load_raw('$rx'); \
          plot_specgram(s, 8000, 200, 3000); print('${filename}_spec.jpg', '-djpg'); \
          quit" | octave-cli -p ${CODEC2_PATH}/octave -qf > /dev/null

    # extract chirp at start and estimate C/No, and chirp start time.  We allow a 10 second window
    rx_chirp=$(mktemp)
    sox $rx -t .s16 ${rx_chirp}.raw trim 0 10
    cat  ${rx_chirp}.raw | python3 int16tof32.py --zeropad > ${rx_chirp}.f32
    est_log=$(mktemp)
    python3 est_CNo.py ${rx_chirp}.f32 | tee $est_log
    chirp_start=$(cat ${est_log} | grep "Measured:" | tr -s ' ' | cut -d' ' -f2)
    cat ${est_log} | tail -n 2 > ${filename}_report.txt
    echo >> ${filename}_report.txt

    # remove silence before chirp
    rx_trim=$(mktemp).wav
    sox $rx $rx_trim trim $chirp_start
    cp $rx_trim $rx

    # 4 sec chirp - 1 sec sil - x sec SSB - 1 sec sil - x sec rade1 - 1 sec sil - x sec rade 2 - 1 sec sil
    # total = 4 + 3(1+x) + 1
    # start_rade1 = 4+(1+x)
    # start_rade2 = 4+2*(1+x)
    total_duration=$(sox --info -D $rx)
    x=$(python3 -c "x=(${total_duration}-8)/3; print(\"%f\" % x)")
    start_rade1=$(python3 -c "start_rade1=5+${x}+0.5; print(\"%f\" % start_rade1)")
    len_rade1=$(python3 -c "len_rade1=${x}+0.5; print(\"%f\" % len_rade1)")
    start_rade2=$(python3 -c "start_rade2=6+2*${x}+1; print(\"%f\" % start_rade2)")
    rx_rade1=$(mktemp)
    rx_rade2=$(mktemp)
    sox $rx ${filename}_ssb.wav trim 5 $x
    sox $rx -t .s16 ${rx_rade1}.raw trim $start_rade1 $len_rade1
    sox -t .s16 -r 8000 -c 1 ${rx_rade1}.raw rade1_in.wav # wave version for debugging
    sox $rx -t .s16 ${rx_rade2}.raw trim $start_rade2 $x
    sox -t .s16 -r 8000 -c 1 ${rx_rade2}.raw rade2_in.wav # wave version for debugging

    # Use streaming RADE1 Rx
    cat ${rx_rade1}.raw | python3 int16tof32.py --zeropad > ${rx_rade1}.f32
    cat ${rx_rade1}.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 2>>${filename}_report.txt > features_out_rx1.f32
    lpcnet_demo -fargan-synthesis features_out_rx1.f32 - | sox -t .s16 -r 16000 -c 1 - ${filename}_rade1.wav
    
    # extract freq offset est from RADE V1
    rade1_foff_samples=$(mktemp)
    # sed ensures all lines have a leading whitespace
    cat ${filename}_report.txt | grep sync | sed 's/^/  /' | tr -s ' ' | cut -d' ' -f17 > $rade1_foff_samples
    rade1_foff=$(python3 -c "import numpy as np; foff_samples=np.loadtxt(\"${rade1_foff_samples}\"); print(f\"{np.mean(foff_samples)}\") ")

    # RADE2 Rx
    cat ${rx_rade2}.raw | python3 int16tof32.py --zeropad > ${rx_rade2}.f32
    ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725_ft 250725_ml_sync ${rx_rade2}.f32 ${filename}_rade2.wav --latent-dim 56 --w1_dec 128 --agc --freq_offset ${rade1_foff}

    speechfile_no_path_no_ext=$3
    if [ ! ${speechfile_no_path_no_ext} == "" ]; then
      # optional loss measurements
      python3 loss.py ${speechfile_no_path_no_ext}_features_in.f32 ${speechfile_no_path_no_ext}_features_out_tx1.f32 --features_hat2 features_out_rx1.f32 --compare | sed -n '5p'
      python3 loss.py ${speechfile_no_path_no_ext}_features_in.f32 ${speechfile_no_path_no_ext}_features_out_tx2.f32 --features_hat2 features_out_rx2.f32 --compare | sed -n '5p'
    fi
}

function tx_ssb_radio {
    echo "--------------------------------------------------------"
    echo "Transmitting $1 using SSB radio"
    echo "--------------------------------------------------------"

    tx_file=$1

    echo "Tx data signal"
    freq_Hz=$((freq_kHz*1000))
    # TODO - IC7200 switches from USB sound card to mic when I try to set PKTLSB/PKTUSB
    #      - How to select LSB/USB while keeping internal sound card?
    #usb_lsb=$(python3 -c "print('usb') if ${freq_kHz} >= 10000 else print('lsb')")
    #usb_lsb_upper=$(echo ${usb_lsb} | awk '{print toupper($0)}')
    #run_rigctl "\\set_mode PKT${usb_lsb_upper} 0" $model
    run_rigctl "\\set_freq ${freq_Hz}" $model
    run_rigctl "\\set_ptt 1" $model
    if [ `uname` == "Darwin" ]; then
        AUDIODEV="${soundDevice}" play -t raw -b 16 -c 1 -r 8000 -e signed-integer --endian little ${tx_file}
    else
        aplay --device="${soundDevice}" -f S16_LE ${tx_file} 2>/dev/null
    fi
    if [ $? -ne 0 ]; then
        run_rigctl "\\set_ptt 0" $model
        clean_up
        echo "Problem running aplay!"
        echo "  1. Is ${soundDevice} configured as the default sound device in Settings-Sound?"
        echo "  2. Is pavucontrol running?"
        
        exit 1
    fi
    run_rigctl "\\set_ptt 0" $model
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
    -l)
        loss="$2"	
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
    --tx_path)
        tx_path="$2"
        shift
        shift
    ;;
    -t)
        just_tx=1
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

if [ $# -eq 0 ]; then
    print_help
fi

if [ $rxwavefile -eq 1 ]; then
    if [ ! -f $1 ]; then
        echo "Can't find input wave file: ${1}!"
        exit 1
    fi   
    if [ ! $loss == "" ]; then
        speechfile_no_path_no_ext="${loss##*/}" # Removes path
        speechfile_no_path_no_ext="${speechfile_no_path_no_ext%.*}" # Removes extension
    fi
    process_rx $1 $freq_offset $speechfile_no_path_no_ext
    exit 0
fi

speechfile="$1"
if [ ! -f $speechfile ]; then
    echo "Can't find input file: ${speechfile}!"
    exit 1
fi

if [ $just_tx -eq 1 ]; then
    tx_ssb_radio $1
    exit 0
fi

# check format of input speech file
soxi_log=$(mktemp)
soxi $speechfile > $soxi_log
channels=$(cat ${soxi_log} | grep "Channels" | tr -s ' ' | cut -d' ' -f3)
sample_rate=$(cat ${soxi_log} | grep "Sample Rate" | tr -s ' ' | cut -d' ' -f4)
if [ $channels -ne 1 ] || [ $sample_rate -ne 16000 ]; then
    echo "Input speech wave file must be single channel 16000 Hz sample rate"
    exit 1 
fi

# extract speechfile name without path or extension
speechfile_no_path_no_ext="${speechfile##*/}" # Removes path
speechfile_no_path_no_ext="${speechfile_no_path_no_ext%.*}" # Removes extension

# create Tx file ------------------------

# create 400-2000 Hz chirp header used for C/No est.  We generate 4.5s of chirp, to allow for trimming of
# rx wave file - we need >=4 seconds of received chirp for C/No est at Rx

echo "----------------------------------------------------------------"
echo "Creating chirp - compressed SSB - RADE1 - RADE2 wave file tx.wav"
echo "----------------------------------------------------------------"

chirp=$(mktemp)
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
tx_radae1=tx_rade1
tx_radae2=tx_rade2
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

# create modulated radae V1 signal
./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth $speechfile_pad /dev/null --end_of_over --auxdata --EbNodB 100 \
--bottleneck 3 --pilots --cp 0.004 --rate_Fs --tanh --ssb_bpf --write_rx ${tx_radae1}.f32
# save features in/out for later "loss.py" measurments
cp features_in.f32 ${speechfile_no_path_no_ext}_features_in.f32
cp features_out.f32 ${speechfile_no_path_no_ext}_features_out_tx1.f32
# extract real (I) channel
cat ${tx_radae1}.f32 | python3 f32toint16.py --real --scale 16383 > ${tx_radae1}.raw 

# create modulated radae V2 signal
./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth $speechfile_pad /dev/null --rate_Fs --latent-dim 56 --peak --ssb_bpf \
--cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx ${tx_radae2}.f32
# save features in/out for later "loss.py" measurments
cp features_out.f32 ${speechfile_no_path_no_ext}_features_out_tx2.f32
# extract real (I) channel
cat ${tx_radae2}.f32 | python3 f32toint16.py --real --scale 16383 > ${tx_radae2}.raw 

# Make power of both signals the same, by adjusting the levels to meet the setpoint
if [ $peak -eq 1 ]; then
  set_peak $tx_ssb $setpoint_peak
  set_peak ${tx_radae1}.raw $setpoint_peak
  set_peak ${tx_radae2}.raw $setpoint_peak
else
  set_rms $tx_ssb $setpoint_rms
  set_rms ${tx_radae}.raw $setpoint_rms
fi

# insert 1 second of silence between signals
sox -t .s16 -r 8k -c 1 $tx_ssb -t .s16 -r 8k -c 1 ${tx_ssb}_pad.raw pad 1@0
sox -t .s16 -r 8k -c 1 ${tx_radae1}.raw -t .s16 -r 8k -c 1 ${tx_radae1}_pad.raw pad 1
sox -t .s16 -r 8k -c 1 ${tx_radae2}.raw -t .s16 -r 8k -c 1 ${tx_radae2}_pad.raw pad 1 1

# cat signals together so we can send them over a radio at the same time
cat ${chirp}.raw ${tx_ssb}_pad.raw ${tx_radae1}_pad.raw  ${tx_radae2}_pad.raw > ${tx_path}/tx.raw
sox -t .s16 -r 8000 -c 1 ${tx_path}/tx.raw ${tx_path}/tx.wav

# generate a 4MSP .iq8 file suitable for replaying by HackRF (can disable if not using HackRF)
if [ $hackrf -eq 1 ]; then
  ch ${tx_path}/tx.raw - --complexout | tsrc - - 5 -c | tlininterp - tx.iq8 100 -d -f
fi
 
if [ $tx_file -eq 1 ]; then
  echo "Finished OK!"
  exit 0
fi

tx_ssb_radio ${tx_path}/tx.raw

echo "Finished OK!"
