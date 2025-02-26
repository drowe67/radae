#!/bin/bash -x
#
# Compare models by plotting loss v Eq/No curves

# Run through training dataset on each trained model to build loss versus Eq/No curve
function run_model() {
    model=$1
    dim=$2
    epoch=$3
    chan=$4
    shift
    shift
    shift
    shift
    python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --latent-dim ${dim} \
        --epochs 1 --lr 0.003 --lr-decay-factor 0.0001 \
        ~/Downloads/tts_speech_16k_speexdsp.f32 tmp \
        --range_EbNo --plot_EqNo ${model}_${chan} --initial-checkpoint ${model}/checkpoints/checkpoint_epoch_${epoch}.pth $@
}

# TODO automate here; Makefile like behaivour to generate model${model}_loss_EqNodB.txt files if they don't exist
#run_model model05 80                                       # known good reference
#run_model model09 80                                       # should be a copy of 05
#run_model model10 40                                       # first attempt at dim=40
#run_model model11 40  --q_opt                              # tanh applied to |z|
#run_model model12 40  --bottleneck 2  --range_EbNo_start 0 # tanh applied to |z|, --range_EbNo_start 0
#run_model model13 80  --bottleneck 3  --rate_Fs            # tanh applied to |tx|
#run_model model14 80  --bottleneck 3  --rate_Fs            # tanh applied to |tx|
#run_model model17 80  --bottleneck 3  --range_EbNo_start -9                      # mixed rate Rs with tanh applied to |tx|
#run_model model18 40  --bottleneck 3                     # mixed rate Rs with tanh applied to |tx|
#run_model model17_check7 80 --range_EbNo_start -9 --bottleneck 3                # should be repeat of 17
#run_model ~/tmp/240607_radio_ae/model17_check3 80  --bottleneck 3 --range_EbNo_start -9  # 17 with aux emebdded 25 bits/s data
#run_model model19_check3 80 100 awgn --bottleneck 3 --range_EbNo_start -9  --auxdata # RADE V1 model
#run_model 250117_test 80 200 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata --txbpf # 0dB PAPR, 99% power BW < 1.5Rq, 3 stage clip/filter
#run_model model19_check3 80 100 mpp --bottleneck 3 --range_EbNo_start -6  --auxdata --h_file h_nc20_train_mpp.f32  # RADE V1 model
#run_model 250117_test 80 200 mpp --bottleneck 3 --range_EbNo_start 0 --auxdata --txbpf --h_file h_nc20_train_mpp.f32  # 0dB PAPR, 99% power BW < 1.5Rq, 3 stage clip/filter
#run_model 250204_test 80 200 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata  # trained with complex h (pilotless)
#run_model 250204_test 80 200 mpp --bottleneck 3 --range_EbNo_start -6 --auxdata --h_file h_nc20_train_mpp.c64 --h_complex  # trained with complex h (pilotless)
#run_model 250206_test 120 100 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata  # trained with complex h (pilotless)
#run_model 250206_test 120 100 mpp --bottleneck 3 --range_EbNo_start -6 --auxdata --h_file h_nc30_mpp_test.c64 --h_complex  # trained with complex h (pilotless)
#run_model 250207_test 120 200 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata  # trained with complex h (pilotless)
#run_model 250212_test 120 200 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata  --pilots2 # trained with complex h (pilotless)
#run_model 250213a_test 120 200 awgn --bottleneck 2 --range_EbNo_start -6 --auxdata  # trained with complex h (pilotless)
#run_model 250213_test 120 200 mpp --bottleneck 3 --range_EbNo_start -6 --auxdata --h_file h_nc30_mpp_test.c64 --h_complex  # trained with complex h (pilotless)
#run_model 250225_test 120 200 awgn --bottleneck 3 --range_EbNo_start -9 --auxdata  --pilots2 # trained with complex h (pilotless)

model_list='model19_check3_awgn 250213_test_awgn 250213a_test_awgn 250225_test_awgn'
model_dim=(80 120 120 120)
declare -a model_legend=("RADE V1 AWGN d=80" "250213 AWGN d=120 b3" "250213a AWGN d=120 b2" "250225 AWGN d=120 b3 p2")

loss_EqNo=""
loss_CNo="50,1"
loss_SNR3k="50,3000"
i=0;
for model in $model_list
  do
    loss_EqNo="${loss_EqNo},'${model}_loss_EqNodB.txt','${model_legend[i]}'"
    CNo=",'${model}_loss_EqNodB.txt',${model_dim[i]},'${model_legend[i]}'"
    loss_CNo="${loss_CNo}${CNo}"
    loss_SNR3k="${loss_SNR3k}${CNo}"
    ((i++))
  done
echo "radae_plots; loss_EqNo_plot('loss_EqNo_models',''${loss_EqNo}); quit" | octave-cli -qf # PNG
echo "radae_plots; loss_EqNo_plot('','loss_EqNo_models'${loss_EqNo}); quit" | octave-cli -qf # EPS
echo "radae_plots; loss_CNo_plot('loss_CNo_models','',${loss_CNo}); quit" | octave-cli -qf # PNG
echo "radae_plots; loss_CNo_plot('loss_SNR3k_models','',${loss_SNR3k}); quit" | octave-cli -qf # PNG
echo "radae_plots; loss_CNo_plot('','250226_loss_SNR3k_models',${loss_SNR3k}); quit" | octave-cli -qf # EPS
