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

#run_model model05 80                                       # known good reference
#run_model model09 80                                       # should be a copy of 05
#run_model model10 40                                       # first attempt at dim=40
#run_model model11 40  --q_opt                              # tanh applied to |z|
#run_model model12 40  --bottleneck 2  --range_EbNo_start 0 # tanh applied to |z|, --range_EbNo_start 0
#run_model model13 80  --bottleneck 3  --rate_Fs            # tanh applied to |tx|
#run_model model14 80  --bottleneck 3  --rate_Fs            # tanh applied to |tx|


echo "radae_plots; loss_EqNo_plot('loss_EqNo_models',
                                  'model05_loss_EqNodB.txt','m5 dim 80 mpp 1D #1', \
                                  'model09_loss_EqNodB.txt','m9 dim 80 mpp 1D #2', \
                                  'model10_loss_EqNodB.txt','m10 dim 40 awgn 1D', \
                                  'model11_loss_EqNodB.txt','m11 dim 40 awgn 2D #1', \
                                  'model12_loss_EqNodB.txt','m12 dim 40 awgn 2D #2', \
                                  'model13_loss_EqNodB.txt','m13 dim 80 awgn 2D |tx| #1', \
                                  'model14_loss_EqNodB.txt','m14 dim 80 awgn 2D |tx| #2'); \
     quit" | octave-cli -qf
echo "radae_plots; loss_CNo_plot('loss_CNo_models', 50, 1, \
                                 'model05_loss_EqNodB.txt',80,'m5 dim 80 mpp 1D #1', \
                                 'model10_loss_EqNodB.txt',40,'m10 dim 40 awgn 1D', \
                                 'model12_loss_EqNodB.txt',40,'m12 dim 40 awgn 2D #2', \
                                 'model13_loss_EqNodB.txt',80,'m13 dim 80 awgn 2D |tx| #1', \
                                 'model14_loss_EqNodB.txt',80,'m14 dim 80 awgn 2D |tx| #2'); \
     quit" | octave-cli -qf
echo "radae_plots; loss_CNo_plot('loss_SNR_models', 50, 3000, \
                                 'model05_loss_EqNodB.txt',80,'m5 dim 80 mpp 1D #1', \
                                 'model10_loss_EqNodB.txt',40,'m10 dim 40 awgn 1D', \
                                 'model12_loss_EqNodB.txt',40,'m12 dim 40 awgn 2D #2',
                                 'model13_loss_EqNodB.txt',80,'m13 dim 80 awgn 2D |tx| #1', \
                                 'model14_loss_EqNodB.txt',80,'m14 dim 80 awgn 2D |tx| #2'); \
     quit" | octave-cli -qf
