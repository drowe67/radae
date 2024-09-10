#!/bin/bash -x
#
# Analysing samples from Rick, W7YC, that have a mechanical artefact when passed through RADAE

results_dir=rick_240910

OPUS=build/src
PATH=${PATH}:${OPUS}

function run_tests {
  wav_in=$1
  filename=$(basename -- "$wav_in")
  filename="${filename%.*}"
  results=$2
  start=$3
  finish=$4
  mkdir -p ${results}
  tmp=$(mktemp)

  sox ${wav_in} ${results}/${filename}_01.wav trim ${start} ${finish}

  # 1. vanilla FARGAN vocoder (no RADAE)
  sox $wav_in -t .s16 -r 16000 -c 1 - | \
  lpcnet_demo -features - - | \
  lpcnet_demo -fargan-synthesis - - | \
  sox -t .s16 -r 16000 -c 1 - ${results}/${filename}_02_fargan.wav trim ${start} ${finish}

  # 2. FARGAN+RADAE model05 (generates features_out.f32)
  ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth ${results}/${filename}_01.wav ${results}/${filename}_03_radae.wav
  mv features_out.f32 features_out_model05.f32

  # 3. FARGAN+RADAE model_bbfm_01 (generates features_out.f32)
  ./inference.sh  model_bbfm_01/checkpoints/checkpoint_epoch_100.pth ${results}/${filename}_01.wav ${results}/${filename}_04_radae.wav
  mv features_out.f32 features_out_model_bbfm_01.f32
  
  # Plot the pitch and voicing tracks from to see if RADAE has messed anything up 
  echo "radae_plots; 
        plot_sample_spec('${wav_in}','${results}/${filename}_10_spec.png'); \
        compare_pitch_corr('${wav_in}','features_in.f32','features_out_model05.f32','${results}/${filename}_11_feat.png'); \
        compare_pitch_corr('${wav_in}','features_in.f32','features_out_model_bbfm_01.f32','${results}/${filename}_12_feat.png'); \
        quit;" | octave-cli -qf
}

date > ${results_dir}/zz_README.txt
cat >> ${results_dir}/zz_README.txt << 'endreadme'
01 - original
02 - Vanilla FARGAN
03 - FARGAN + RADAE model05
04 - FARGAN + RADAE model_bbfm_01
10 - spectrum of entire sample
11 - compare FARGAN/RADAE pitch & voicing for 5s of model05 
12 - compare FARGAN/RADAE pitch & voicing for 5s of model_bbfm_01

Notes:
* 03 and 04 tested with no noise on channel
* model05 - OFDM channels trained for range of SNRs from very low to medium (moderate average loss)
* model_bbfm_01 - baseband FM channels, trained for high SNRs (low average loss)
endreadme

# TODO:
# 1. try different models - do they all mess up the speech?
# 2. try different speakers - any obvious differences in features, e.g. spectra/voicing/pitch
# model05 used for convenience, useful control and perfoms the same as model19_check3

run_tests wav/brian_g8sez.wav ${results_dir} 3 10
run_tests wav/rick_w7yc_1.wav ${results_dir} 0.5 7
run_tests wav/rick_w7yc_2.wav ${results_dir} 0 5
