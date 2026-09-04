"""
Export RADE V2 encoder, decoder and frame-sync weights to C source files
suitable for use in the radae_nopy C port.

Usage:
    python3 export_rade_v2_weights.py <model_checkpoint> <ml_sync_model> <output_dir>

Example:
    python3 export_rade_v2_weights.py 250725/checkpoints/checkpoint_epoch_200.pth \
        250725a_ml_sync src_v2

The script produces:
    rade_enc_v2_data.{h,c}   -- CoreEncoderStatefull weights
    rade_dec_v2_data.{h,c}   -- CoreDecoderStatefull weights
    rade_sync_data.{h,c}     -- FrameSyncNet weights
    rade_v2_constants.h      -- V2 constants (latent_dim, layer sizes, etc.)

Copyright (c) 2025 David Rowe
Two-clause BSD license.
"""

import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'weight-exchange'))

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint',    type=str, help='RADE V2 model checkpoint (.pth)')
parser.add_argument('ml_sync_model', type=str, help='ML frame-sync model file')
parser.add_argument('output_dir',    type=str, help='output folder for generated C files')
parser.add_argument('--latent-dim',  type=int, default=56,  help='latent dimension (default: 56)')
parser.add_argument('--w1-dec',      type=int, default=128, help='decoder hidden width (default: 128)')
parser.add_argument('--noauxdata',   dest='auxdata', action='store_false',
                    help='disable auxiliary data symbol (use 20 features instead of 21)')
parser.set_defaults(auxdata=True)
args = parser.parse_args()

import torch
import numpy as np

from radae import RADAE
from models_sync import FrameSyncNet
from wexchange.torch import dump_torch_weights
from wexchange.c_export import CWriter, print_vector


def c_export(args, model, frame_sync_nn):

    message = f"Auto generated from checkpoint {os.path.basename(args.checkpoint)}"

    enc_writer      = CWriter(os.path.join(args.output_dir, "rade_enc_v2_data"),
                              message=message, model_struct_name='RADEEncV2')
    dec_writer      = CWriter(os.path.join(args.output_dir, "rade_dec_v2_data"),
                              message=message, model_struct_name='RADEDecV2')
    sync_writer     = CWriter(os.path.join(args.output_dir, "rade_sync_data"),
                              message=message, model_struct_name='RADESync')
    constants_writer = CWriter(os.path.join(args.output_dir, "rade_v2_constants"),
                               message=message, header_only=True, enable_binary_blob=False)

    # common includes for enc/dec/sync writers
    for writer in [enc_writer, dec_writer, sync_writer]:
        writer.header.write(
"""
#include "opus_types.h"

#include "rade_v2_core.h"

#include "rade_v2_constants.h"

"""
        )

    # -------------------------------------------------------------------------
    # Encoder  (CoreEncoderStatefull)
    # Layer widths: dense_1=64, GRUs=64, convs=96, z_dense=latent_dim
    # -------------------------------------------------------------------------
    enc_base = 'core_encoder_statefull.module'

    encoder_dense_layers = [
        (f'{enc_base}.dense_1',  'enc_v2_dense1',  False),
        (f'{enc_base}.z_dense',  'enc_v2_zdense',  False),
    ]
    for path, export_name, quantize in encoder_dense_layers:
        layer = model.get_submodule(path)
        dump_torch_weights(enc_writer, layer, name=export_name, verbose=True,
                           quantize=quantize, scale=None)

    # GRUStatefull wraps nn.GRU as .gru
    encoder_gru_layers = [
        (f'{enc_base}.gru1.gru', 'enc_v2_gru1', True),
        (f'{enc_base}.gru2.gru', 'enc_v2_gru2', True),
        (f'{enc_base}.gru3.gru', 'enc_v2_gru3', True),
        (f'{enc_base}.gru4.gru', 'enc_v2_gru4', True),
        (f'{enc_base}.gru5.gru', 'enc_v2_gru5', True),
    ]
    enc_max_rnn_units = max([
        dump_torch_weights(enc_writer, model.get_submodule(path), export_name,
                           verbose=True, input_sparse=True, quantize=quantize,
                           scale=None, recurrent_scale=None)
        for path, export_name, quantize in encoder_gru_layers
    ])

    # Conv1DStatefull wraps nn.Conv1d as .conv
    encoder_conv_layers = [
        (f'{enc_base}.conv1.conv', 'enc_v2_conv1', True),
        (f'{enc_base}.conv2.conv', 'enc_v2_conv2', True),
        (f'{enc_base}.conv3.conv', 'enc_v2_conv3', True),
        (f'{enc_base}.conv4.conv', 'enc_v2_conv4', True),
        (f'{enc_base}.conv5.conv', 'enc_v2_conv5', True),
    ]
    enc_max_conv_inputs = max([
        dump_torch_weights(enc_writer, model.get_submodule(path), export_name,
                           verbose=True, quantize=quantize, scale=None)
        for path, export_name, quantize in encoder_conv_layers
    ])

    del enc_writer

    # -------------------------------------------------------------------------
    # Decoder  (CoreDecoderStatefull)
    # Layer widths: dense_1=w1_dec, GRUs=w1_dec, convs=32, output=frames*features
    # -------------------------------------------------------------------------
    dec_base = 'core_decoder_statefull.module'

    decoder_dense_layers = [
        (f'{dec_base}.dense_1',    'dec_v2_dense1',  False),
        (f'{dec_base}.glu1.gate',  'dec_v2_glu1',    True),
        (f'{dec_base}.glu2.gate',  'dec_v2_glu2',    True),
        (f'{dec_base}.glu3.gate',  'dec_v2_glu3',    True),
        (f'{dec_base}.glu4.gate',  'dec_v2_glu4',    True),
        (f'{dec_base}.glu5.gate',  'dec_v2_glu5',    True),
        (f'{dec_base}.output',     'dec_v2_output',  False),
    ]
    for path, export_name, quantize in decoder_dense_layers:
        layer = model.get_submodule(path)
        dump_torch_weights(dec_writer, layer, name=export_name, verbose=True,
                           quantize=quantize, scale=None)

    decoder_gru_layers = [
        (f'{dec_base}.gru1.gru', 'dec_v2_gru1', True),
        (f'{dec_base}.gru2.gru', 'dec_v2_gru2', True),
        (f'{dec_base}.gru3.gru', 'dec_v2_gru3', True),
        (f'{dec_base}.gru4.gru', 'dec_v2_gru4', True),
        (f'{dec_base}.gru5.gru', 'dec_v2_gru5', True),
    ]
    dec_max_rnn_units = max([
        dump_torch_weights(dec_writer, model.get_submodule(path), export_name,
                           verbose=True, input_sparse=True, quantize=quantize,
                           scale=None, recurrent_scale=None)
        for path, export_name, quantize in decoder_gru_layers
    ])

    decoder_conv_layers = [
        (f'{dec_base}.conv1.conv', 'dec_v2_conv1', True),
        (f'{dec_base}.conv2.conv', 'dec_v2_conv2', True),
        (f'{dec_base}.conv3.conv', 'dec_v2_conv3', True),
        (f'{dec_base}.conv4.conv', 'dec_v2_conv4', True),
        (f'{dec_base}.conv5.conv', 'dec_v2_conv5', True),
    ]
    dec_max_conv_inputs = max([
        dump_torch_weights(dec_writer, model.get_submodule(path), export_name,
                           verbose=True, quantize=quantize, scale=None)
        for path, export_name, quantize in decoder_conv_layers
    ])

    del dec_writer

    # -------------------------------------------------------------------------
    # FrameSyncNet: three dense layers (56->64 ReLU, 64->64 ReLU, 64->1 Sigmoid)
    # -------------------------------------------------------------------------
    sync_dense_layers = [
        ('linear_stack.0', 'sync_dense1', False),
        ('linear_stack.2', 'sync_dense2', False),
        ('linear_stack.4', 'sync_dense3', False),
    ]
    for path, export_name, quantize in sync_dense_layers:
        layer = frame_sync_nn.get_submodule(path)
        dump_torch_weights(sync_writer, layer, name=export_name, verbose=True,
                           quantize=quantize, scale=None)

    del sync_writer

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    num_features = model.feature_dim
    constants_writer.header.write(
f"""
#define RADE_V2_FRAMES_PER_STEP {model.enc_stride}

#define RADE_V2_LATENT_DIM {args.latent_dim}

#define RADE_V2_NUM_FEATURES {num_features}

#define RADE_V2_ENC_MAX_RNN_NEURONS {enc_max_rnn_units}

#define RADE_V2_ENC_MAX_CONV_INPUTS {enc_max_conv_inputs}

#define RADE_V2_DEC_MAX_RNN_NEURONS {dec_max_rnn_units}

#define RADE_V2_DEC_MAX_CONV_INPUTS {dec_max_conv_inputs}

"""
    )
    del constants_writer


if __name__ == "__main__":

    num_features = 21 if args.auxdata else 20
    os.makedirs(args.output_dir, exist_ok=True)

    # Load RADE V2 model using the same pattern as rx2.py:
    #   1. Instantiate with w1_dec=w1_dec_stateful=args.w1_dec
    #   2. Filter mismatched checkpoint entries (core_decoder_statefull was trained with default w1=96)
    #   3. Copy core_decoder -> core_decoder_statefull via load_state_dict helper
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = RADAE(num_features, args.latent_dim, EbNodB=100, Nzmf=1,
                  rate_Fs=True, cyclic_prefix=0.004,
                  w1_dec=args.w1_dec, w1_dec_stateful=args.w1_dec, peak=True)

    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # Copy core_encoder -> core_encoder_statefull and core_decoder -> core_decoder_statefull
    model.core_encoder_statefull_load_state_dict()
    model.core_decoder_statefull_load_state_dict()

    # Remove weight_norm (applied to GLU gate layers)
    def _remove_weight_norm(m):
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            return
    model.apply(_remove_weight_norm)

    # Load FrameSyncNet
    frame_sync_nn = FrameSyncNet(args.latent_dim)
    frame_sync_nn.load_state_dict(torch.load(args.ml_sync_model, weights_only=True))
    frame_sync_nn.eval()

    c_export(args, model, frame_sync_nn)
