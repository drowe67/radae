#!/bin/bash -x
#
# Compare models by plotting loss v Eq/No curves

# Run trough training dataset on each trained model to build loss versus Eq/No curve
function run_model() {
    model=$1
    dim=$2
    shift
    shift
    python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --latent-dim ${dim} \
        --epochs 1 --lr 0.003 --lr-decay-factor 0.0001 \
        ~/Downloads/tts_speech_16k_speexdsp.f32 tmp \
        --range_EbNo --plot_EqNo ${model} --initial-checkpoint ${model}/checkpoints/checkpoint_epoch_100.pth $@
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

model_list='05 14 17 18'
model_dim=(80 80 80 40)
declare -a model_legend=("m05 dim 80 mpp 1D #1" \
                         "m14 dim 80 mpp 2D Fs |tx|" \
                         "m17 dim 80 mpp 2D |tx| mixed" \
                         "m18 dim 40 mpp 2D |tx| mixed")

loss_EqNo="'loss_EqNo_models'"
loss_CNo="'loss_CNo_models',50,1"
loss_SNR3k="'loss_SNR3k_models',50,3000"
i=0;
for model in $model_list
  do
    loss_EqNo="${loss_EqNo},'model${model}_loss_EqNodB.txt','${model_legend[i]}'"
    CNo=",'model${model}_loss_EqNodB.txt',${model_dim[i]},'${model_legend[i]}'"
    loss_CNo="${loss_CNo}${CNo}"
    loss_SNR3k="${loss_SNR3k}${CNo}"
    ((i++))
  done
echo "radae_plots; loss_EqNo_plot(${loss_EqNo}); quit" | octave-cli -qf
echo "radae_plots; loss_CNo_plot(${loss_CNo}); quit" | octave-cli -qf
echo "radae_plots; loss_CNo_plot(${loss_SNR3k}); quit" | octave-cli -qf
