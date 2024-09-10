#!/bin/bash -x
#
# Analysing samples from Rick, W7YC, that have a mechanical artefact when passed through RADAE

results_dir=rick_240908

OPUS=build/src
PATH=${PATH}:${OPUS}

function run_tests {
  wav_in=$1
  filename=$(basename -- "$wav_in")
  filename="${filename%.*}"
  model=$2
  results=$3
  mkdir -p ${results}

  cp ${wav_in} ${results}/${filename}_01.wav

  # 1. vanilla FARGAN vocoder (no RADAE)
  sox $wav_in -t .s16 -r 16000 -c 1 - | \
  lpcnet_demo -features - - | \
  lpcnet_demo -fargan-synthesis - - | \
  sox -t .s16 -r 16000 -c 1 - ${results}/${filename}_02_fargan.wav

  # 2. FARGAN+RADAE (generates features_out.f32)
  ./inference.sh $2 ${wav_in} ${results}/${filename}_03_radae.wav

  # lets plot the pitch and voicing tracks to see if RADAE has messed anything up 
  echo "radae_plots; rick_compare('${wav_in}','features_in.f32','features_out.f32','${results}/${filename}_04_feat.png','${results}/${filename}_05_spec.png'); quit;" | octave-cli -qf
}

cat > ${results_dir}/zz_README.txt <<'endreadme'
01 - original
02 - Vanilla FARGAN
03 - FARGAN + RADAE
04 - plot of selected features for first 5s
05 - spectrum of entire sample
endreadme

# TODO:
# 1. try different models - do they all mess up the speech?
# 2. try different speakers - any obvious differences in features, e.g. spectra/voicing/pitch
# model05 used for convenience, useful control and perfoms the same as model19_check3

run_tests wav/rick_w7yc_1.wav model05/checkpoints/checkpoint_epoch_100.pth ${results_dir}
run_tests wav/rick_w7yc_2.wav model05/checkpoints/checkpoint_epoch_100.pth rick_240908
run_tests wav/brian_g8sez.wav model05/checkpoints/checkpoint_epoch_100.pth rick_240908
