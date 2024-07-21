#!/bin/bash -x
#
# Build libopus.a from previously built objects

OPUS_DIR=$1
cd $OPUS_DIR
ar rcs libopus.a dnn/.libs/burg.o dnn/.libs/freq.o dnn/.libs/fargan.o dnn/.libs/fargan_data.o dnn/.libs/lpcnet_enc.o dnn/.libs/lpcnet_plc.o dnn/.libs/lpcnet_tables.o dnn/.libs/nnet.o dnn/.libs/nnet_default.o dnn/.libs/plc_data.o dnn/.libs/parse_lpcnet_weights.o dnn/.libs/pitchdnn.o dnn/.libs/pitchdnn_data.o dnn/.libs/dred_rdovae_enc.o dnn/.libs/dred_rdovae_enc_data.o dnn/.libs/dred_rdovae_dec.o dnn/.libs/dred_rdovae_dec_data.o dnn/.libs/dred_rdovae_stats_data.o silk/.libs/dred_encoder.o silk/.libs/dred_coding.o silk/.libs/dred_decoder.o dnn/x86/.libs/x86_dnn_map.o dnn/x86/.libs/nnet_sse2.o dnn/x86/.libs/nnet_sse4_1.o dnn/x86/.libs/nnet_avx2.o celt/.libs/bands.o celt/.libs/celt.o celt/.libs/celt_encoder.o celt/.libs/celt_decoder.o celt/.libs/cwrs.o celt/.libs/entcode.o celt/.libs/entdec.o celt/.libs/entenc.o celt/.libs/kiss_fft.o celt/.libs/laplace.o celt/.libs/mathops.o celt/.libs/mdct.o celt/.libs/modes.o celt/.libs/pitch.o celt/.libs/celt_lpc.o celt/.libs/quant_bands.o celt/.libs/rate.o celt/.libs/vq.o celt/x86/.libs/x86cpu.o celt/x86/.libs/x86_celt_map.o celt/x86/.libs/pitch_sse.o celt/x86/.libs/pitch_sse2.o celt/x86/.libs/vq_sse2.o celt/x86/.libs/celt_lpc_sse4_1.o celt/x86/.libs/pitch_sse4_1.o celt/x86/.libs/pitch_avx.o
