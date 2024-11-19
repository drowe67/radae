"""
/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'weight-exchange'))

parser = argparse.ArgumentParser()

parser.add_argument('checkpoint', type=str, help='model checkpoint')
parser.add_argument('output_dir', type=str, help='output folder')
parser.add_argument('--format', choices=['C', 'numpy'], help='output format, default: C', default='C')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--noauxdata', dest="auxdata", action='store_false', help='disable injection of auxillary data symbols')
parser.set_defaults(auxdata=True)
args = parser.parse_args()

import torch
import numpy as np

from radae import RADAE
from wexchange.torch import dump_torch_weights
from wexchange.c_export import CWriter, print_vector


def c_export(args, model):

    message = f"Auto generated from checkpoint {os.path.basename(args.checkpoint)}"

    enc_writer = CWriter(os.path.join(args.output_dir, "rade_enc_data"), message=message, model_struct_name='RADEEnc')
    dec_writer = CWriter(os.path.join(args.output_dir, "rade_dec_data"), message=message, model_struct_name='RADEDec')
    #stats_writer = CWriter(os.path.join(args.output_dir, "rade_stats_data"), message=message, enable_binary_blob=False)
    constants_writer = CWriter(os.path.join(args.output_dir, "rade_constants"), message=message, header_only=True, enable_binary_blob=False)

    # some custom includes
    for writer in [enc_writer, dec_writer]:
        writer.header.write(
f"""
#include "opus_types.h"

#include "rade_core.h"

#include "rade_constants.h"

"""
        )


    # encoder
    encoder_dense_layers = [
        ('core_encoder.module.dense_1'       , 'enc_dense1',   'TANH', False,),
        ('core_encoder.module.z_dense'       , 'enc_zdense',   'LINEAR', False,)
    ]

    for name, export_name, _, quantize in encoder_dense_layers:
        layer = model.get_submodule(name)
        dump_torch_weights(enc_writer, layer, name=export_name, verbose=True, quantize=quantize, scale=None)


    encoder_gru_layers = [
        ('core_encoder.module.gru1'       , 'enc_gru1',   'TANH', True),
        ('core_encoder.module.gru2'       , 'enc_gru2',   'TANH', True),
        ('core_encoder.module.gru3'       , 'enc_gru3',   'TANH', True),
        ('core_encoder.module.gru4'       , 'enc_gru4',   'TANH', True),
        ('core_encoder.module.gru5'       , 'enc_gru5',   'TANH', True),
    ]

    enc_max_rnn_units = max([dump_torch_weights(enc_writer, model.get_submodule(name), export_name, verbose=True, input_sparse=True, quantize=quantize, scale=None, recurrent_scale=None)
                             for name, export_name, _, quantize in encoder_gru_layers])


    encoder_conv_layers = [
        ('core_encoder.module.conv1.conv'       , 'enc_conv1',   'TANH', True),
        ('core_encoder.module.conv2.conv'       , 'enc_conv2',   'TANH', True),
        ('core_encoder.module.conv3.conv'       , 'enc_conv3',   'TANH', True),
        ('core_encoder.module.conv4.conv'       , 'enc_conv4',   'TANH', True),
        ('core_encoder.module.conv5.conv'       , 'enc_conv5',   'TANH', True),
    ]

    enc_max_conv_inputs = max([dump_torch_weights(enc_writer, model.get_submodule(name), export_name, verbose=True, quantize=quantize, scale=None) for name, export_name, _, quantize in encoder_conv_layers])


    del enc_writer

    # decoder
    decoder_dense_layers = [
        ('core_decoder.module.dense_1'      , 'dec_dense1',  'TANH', False),
        ('core_decoder.module.glu1.gate'    , 'dec_glu1',    'TANH', True),
        ('core_decoder.module.glu2.gate'    , 'dec_glu2',    'TANH', True),
        ('core_decoder.module.glu3.gate'    , 'dec_glu3',    'TANH', True),
        ('core_decoder.module.glu4.gate'    , 'dec_glu4',    'TANH', True),
        ('core_decoder.module.glu5.gate'    , 'dec_glu5',    'TANH', True),
        ('core_decoder.module.output'       , 'dec_output',  'LINEAR', False)
    ]

    for name, export_name, _, quantize in decoder_dense_layers:
        layer = model.get_submodule(name)
        dump_torch_weights(dec_writer, layer, name=export_name, verbose=True, quantize=quantize, scale=None)


    decoder_gru_layers = [
        ('core_decoder.module.gru1'         , 'dec_gru1',    'TANH', True),
        ('core_decoder.module.gru2'         , 'dec_gru2',    'TANH', True),
        ('core_decoder.module.gru3'         , 'dec_gru3',    'TANH', True),
        ('core_decoder.module.gru4'         , 'dec_gru4',    'TANH', True),
        ('core_decoder.module.gru5'         , 'dec_gru5',    'TANH', True),
    ]

    dec_max_rnn_units = max([dump_torch_weights(dec_writer, model.get_submodule(name), export_name, verbose=True, input_sparse=True, quantize=quantize, scale=None, recurrent_scale=None)
                             for name, export_name, _, quantize in decoder_gru_layers])

    decoder_conv_layers = [
        ('core_decoder.module.conv1.conv'       , 'dec_conv1',   'TANH', True),
        ('core_decoder.module.conv2.conv'       , 'dec_conv2',   'TANH', True),
        ('core_decoder.module.conv3.conv'       , 'dec_conv3',   'TANH', True),
        ('core_decoder.module.conv4.conv'       , 'dec_conv4',   'TANH', True),
        ('core_decoder.module.conv5.conv'       , 'dec_conv5',   'TANH', True),
    ]

    dec_max_conv_inputs = max([dump_torch_weights(dec_writer, model.get_submodule(name), export_name, verbose=True, quantize=quantize, scale=None) for name, export_name, _, quantize in decoder_conv_layers])

    del dec_writer

    #del stats_writer

    # constants
    constants_writer.header.write(
f"""
#define RADE_FRAMES_PER_STEP {model.enc_stride}

#define RADE_LATENT_DIM {args.latent_dim}

#define RADE_MAX_RNN_NEURONS {max(enc_max_rnn_units, dec_max_rnn_units)}

#define RADE_MAX_CONV_INPUTS {max(enc_max_conv_inputs, dec_max_conv_inputs)}

#define RADE_ENC_MAX_RNN_NEURONS {enc_max_conv_inputs}

#define RADE_ENC_MAX_CONV_INPUTS {enc_max_conv_inputs}

#define RADE_DEC_MAX_RNN_NEURONS {dec_max_rnn_units}

"""
    )

    del constants_writer


def numpy_export(args, model):

    exchange_name_to_name = {
        'encoder_stack_layer1_dense'    : 'core_encoder.module.dense_1',
        'encoder_stack_layer3_dense'    : 'core_encoder.module.dense_2',
        'encoder_stack_layer5_dense'    : 'core_encoder.module.dense_3',
        'encoder_stack_layer7_dense'    : 'core_encoder.module.dense_4',
        'encoder_stack_layer8_dense'    : 'core_encoder.module.dense_5',
        'encoder_state_layer1_dense'    : 'core_encoder.module.state_dense_1',
        'encoder_state_layer2_dense'    : 'core_encoder.module.state_dense_2',
        'encoder_stack_layer2_gru'      : 'core_encoder.module.gru_1',
        'encoder_stack_layer4_gru'      : 'core_encoder.module.gru_2',
        'encoder_stack_layer6_gru'      : 'core_encoder.module.gru_3',
        'encoder_stack_layer9_conv'     : 'core_encoder.module.conv1',
        'statistical_model_embedding'   : 'statistical_model.quant_embedding',
        'decoder_state1_dense'          : 'core_decoder.module.gru_1_init',
        'decoder_state2_dense'          : 'core_decoder.module.gru_2_init',
        'decoder_state3_dense'          : 'core_decoder.module.gru_3_init',
        'decoder_stack_layer1_dense'    : 'core_decoder.module.dense_1',
        'decoder_stack_layer3_dense'    : 'core_decoder.module.dense_2',
        'decoder_stack_layer5_dense'    : 'core_decoder.module.dense_3',
        'decoder_stack_layer7_dense'    : 'core_decoder.module.dense_4',
        'decoder_stack_layer8_dense'    : 'core_decoder.module.dense_5',
        'decoder_stack_layer9_dense'    : 'core_decoder.module.output',
        'decoder_stack_layer2_gru'      : 'core_decoder.module.gru_1',
        'decoder_stack_layer4_gru'      : 'core_decoder.module.gru_2',
        'decoder_stack_layer6_gru'      : 'core_decoder.module.gru_3'
    }

    name_to_exchange_name = {value : key for key, value in exchange_name_to_name.items()}

    for name, exchange_name in name_to_exchange_name.items():
        print(f"printing layer {name}...")
        dump_torch_weights(os.path.join(args.output_dir, exchange_name), model.get_submodule(name))


if __name__ == "__main__":

    num_features = 20
    if args.auxdata:
        num_features += 1
    os.makedirs(args.output_dir, exist_ok=True)

    # load model from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # Note only a few parms required to extract weights for core encoder/decoder.  The weights are not affected
    # by the "bottleneck" and rate Rs or rate Fs operation. For example this script can be used for model05 and model19_check3
    model = RADAE(num_features, args.latent_dim, EbNodB=100)
    model.load_state_dict(checkpoint['state_dict'], strict=False, weights_only=True)

    def _remove_weight_norm(m):
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:  # this module didn't have weight norm
            return
    model.apply(_remove_weight_norm)

    if args.format == 'C':
        c_export(args, model)
    elif args.format == 'numpy':
        numpy_export(args, model)
    else:
        raise ValueError(f'error: unknown export format {args.format}')
