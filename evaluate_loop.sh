#!/bin/bash -x
#
# Runs evaluate.sh over a bunch of Eb/No set points.

function print_help {
    echo "usage:"
    echo "  ./evaluate_loop.sh model_number sample.[s16|wav] out_dir"
    echo ""
    echo "example usage:"
    echo "  ./evaluate_loop.sh model17 wav/peter.wav 240524"
     exit 1
}

function evaluate_single {
    model=$1
    fullfile=$2
    out_dir=$3
    EbNodB=$4
    shift; shift; shift; shift
    
    case $model in
    model17)
        ./evaluate.sh model17/checkpoints/checkpoint_epoch_100.pth $fullfile $out_dir $EbNodB --bottleneck 3 -d --peak $@
    ;;
    model18)
        ./evaluate.sh model18/checkpoints/checkpoint_epoch_100.pth $fullfile $out_dir $EbNodB --bottleneck 3 -d --peak --latent_dim 40 $@
    ;;
    *)
        echo "unknown model"
        exit 0
    ;;
    esac
}

if [ $# -lt 3 ]; then
    print_help
fi

model=$1
fullfile=$2
out_dir=$3

EbNodB_awgn="0 3 6 100"
for EbNodB in $EbNodB_awgn
do
    evaluate_single $model $fullfile $out_dir $EbNodB
done

EbNodB_mpp="3 6 16 26"

for EbNodB in $EbNodB_mpp
do
    evaluate_single $model $fullfile $out_dir $EbNodB --g_file g_mpp.f32
done

