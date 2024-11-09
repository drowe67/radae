/* RADE core encoder test program, features.f32 on stdin, z.f32 on stdout */

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

    int auxdata = 1;         // TODO: this could be a CLI option, defaults to 1
    int nb_total_features = 36;
    int num_features = 20;
    int num_used_features = 20;
    int frames_per_step = 4; // TODO dig this out of network somehow

    if (auxdata) {
        num_features += 1;
    }

    if (init_radeenc(&enc_model, rdovaeenc_arrays) != 0) {
        fprintf(stderr, "Error initialising encoder model\n");
        exit(1);        
    }
    rade_init_encoder(&enc_state);

    int n_features_in = enc_model.enc_dense1.nb_inputs;
    assert(enc_model.enc_zdense.nb_outputs == RADE_LATENT_DIM);
    fprintf(stderr, "n_features_in: %d n_z_out: %d", n_features_in, enc_model.enc_zdense.nb_outputs);

    float features_read[frames_per_step*nb_total_features];
    float features[n_features_in];
    float z[RADE_LATENT_DIM];
    int arch = opus_select_arch();

    while((size_t)frames_per_step*nb_total_features == fread(features_read, sizeof(float), frames_per_step*nb_total_features, stdin)) {
        for (int i=0; i<frames_per_step; i++) {
            for(int j=0; j<num_used_features; j++)
                features[i*num_features+j] = features_read[i*nb_total_features+j];
            if (auxdata)
                features[i*num_features+num_used_features] = 0.0;
        }
        rade_core_encoder(&enc_state, &enc_model, z, features, arch);
        fwrite(z, sizeof(float), RADE_LATENT_DIM, stdout);
        fflush(stdout);
    }

    return 0;
}
