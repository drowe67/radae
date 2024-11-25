/* Writes the compiled-in core encoder and decoder weights to a binary
   blob file, that can be used later to run-time init a model. */

/* Copyright (c) 2023 Amazon */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "nnet.h"
#include "os_support.h"
#include "arch.h"

/* This is a bit of a hack because we need to build nnet_data.c and plc_data.c without USE_WEIGHTS_FILE,
   but USE_WEIGHTS_FILE is defined in config.h. */
#undef HAVE_CONFIG_H
#ifdef USE_WEIGHTS_FILE
#undef USE_WEIGHTS_FILE
#endif
#include "rade_enc_data.c"
#include "rade_dec_data.c"

void write_weights(const WeightArray *list, FILE *fout)
{
  int i=0;
  unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};
  while (list[i].name != NULL) {
    WeightHead h;
    if (strlen(list[i].name) >= sizeof(h.name) - 1) {
      printf("[write_weights] warning: name %s too long\n", list[i].name);
    }
    memcpy(h.head, "DNNw", 4);
    h.version = WEIGHT_BLOB_VERSION;
    h.type = list[i].type;
    h.size = list[i].size;
    h.block_size = (h.size+WEIGHT_BLOCK_SIZE-1)/WEIGHT_BLOCK_SIZE*WEIGHT_BLOCK_SIZE;
    OPUS_CLEAR(h.name, sizeof(h.name));
    strncpy(h.name, list[i].name, sizeof(h.name));
    h.name[sizeof(h.name)-1] = 0;
    celt_assert(sizeof(h) == WEIGHT_BLOCK_SIZE);
    fwrite(&h, 1, WEIGHT_BLOCK_SIZE, fout);
    fwrite(list[i].data, 1, h.size, fout);
    fwrite(zeros, 1, h.block_size-h.size, fout);
    i++;
  }
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s weights_blob.bin\n", argv[0]);
    exit(1);
  }
  FILE *fout = fopen(argv[1], "w");
  assert(fout != NULL);
  write_weights(radeenc_arrays, fout);
  write_weights(radedec_arrays, fout);
  fclose(fout);
  return 0;
}
