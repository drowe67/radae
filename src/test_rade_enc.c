#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "rade_core.h" 
#include "rade_enc.h"
#include "rade_enc_data.h"

int opus_select_arch(void);

int main(void)
{
    RADEEnc      enc_model;
    RADEEncState enc_state;

    if (init_radeenc(&enc_model, rdovaeenc_arrays) != 0) {
        fprintf(stderr, "Error initialising encoder model\n");
        exit(1);        
    }
    rade_init_encoder(&enc_state);

    int n_features_in = enc_model.enc_dense1.nb_inputs;
    assert(enc_model.enc_zdense.nb_outputs == RADE_LATENT_DIM);
    fprintf(stderr, "n_features_in: %d n_z_out: %d", n_features_in, enc_model.enc_zdense.nb_outputs);
    float features[n_features_in];
    float z[RADE_LATENT_DIM];
    int arch = opus_select_arch();

    while((size_t)n_features_in == fread(features, sizeof(float), n_features_in, stdin)) {
        rade_core_encoder(&enc_state, &enc_model, z, features, arch);
        fwrite(z, sizeof(float), RADE_LATENT_DIM, stdout);
        fflush(stdout);
    }

    return 0;
}
