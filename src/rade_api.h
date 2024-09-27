/*---------------------------------------------------------------------------*\

  rade_api.h

  Library of API functions that implement the Radio Autoencoder API.

\*---------------------------------------------------------------------------*/

/*
  Copyright (C) 2024 David Rowe

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

#ifndef __RADE_API__
#define __RADE_API__

#include <sys/types.h>

// This declares a single-precision (float) complex number

#ifndef __RADE_COMP__
#define __RADE_COMP__
typedef struct {
  float real;
  float imag;
} RADE_COMP;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Sample rates used
#define RADE_FS_8000 8000           // modem waveform sample rate
#define RADE_FS_16000 16000         // speech sample rate

// note single context only in this version, one context has one Tx, and one Rx
struct rade *rade_open(char model_file[]);
void rade_close(struct *rade);

// Allows API users to determine if the API has changed
int rade_version(void);

// helpers to set up arrays
int rade_n_tx_out(struct *rade);
int rade_n_tx_eoo_out(struct *rade);
int rade_max_nin(struct *rade);
int rade_n_features_in_out(struct *rade);

// Note vocoder is not encapsulated in API in this version
void rade_tx(struct *rade, RADE_COMP tx_out[], float features_in[]);

// call this for the final frame at the end of over
void rade_tx_eoo(struct *rade, RADE_COMP tx_eoo_out[]);

// call me before each call to rade_rx(), provide nin samples to rx_in[]
int rade_nin(struct *rade, void);

// returns non-zero if features[] contains valid output
int rade_rx(struct *rade, float features_out[], RADE_COMP rx_in[]);

// returns non-zero if Rx is currently in sync
int rade_sync(struct *rade);

// returns the current frequency offset of the Rx signal ( when rade_sync()!=0 )
float rade_freq_offset(struct *rade);

#ifdef __cplusplus
}
#endif

#endif  //__RADE_API__
