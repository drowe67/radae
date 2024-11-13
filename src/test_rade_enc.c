/* RADE core encoder test program, features.f32 on stdin, z.f32 on stdout */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "rade_core.h" 
#include "rade_enc.h"
#include "rade_enc_data.h"

int opus_select_arch(void);

int main(int argc, char *argv[])
{
    RADEEnc      enc_model;
    RADEEncState enc_state;

    if (argc < 3) {
        fprintf(stderr, "usage: %s bottleneck[1-3] auxdata[0-1] [weights_blob.bin]\n", argv[0]);
        exit(1);
    }

    int bottleneck = atoi(argv[1]);
    int auxdata = atoi(argv[2]);
    int nb_total_features = 36;
    int num_features = 20;
    int num_used_features = 20;
    int frames_per_step = RADE_FRAMES_PER_STEP;
    
    if (auxdata) {
        num_features += 1;
    }
    int input_dim = num_features*frames_per_step;

    int fd;
    void *data;
    int len;
    int nb_arrays;
    struct stat st;
    WeightArray *list;

    if (argc == 4) {
        const char *filename = argv[3];
        fprintf(stderr, "loading %s ....\n", filename);
        int ret = stat(filename, &st);
        assert(ret != -1);
        len = st.st_size;
        fprintf(stderr, "size is %d\n", len);
        fd = open(filename, O_RDONLY);
        assert(fd != -1);
        // note this needs to stay mapped at run time 
        data = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
        nb_arrays = parse_weights(&list, data, len);
        for (int i=0;i<nb_arrays;i++) {
            fprintf(stderr, "found %s: size %d\n", list[i].name, list[i].size);
        }
        if (init_radeenc(&enc_model, list, input_dim) != 0) {
            fprintf(stderr, "Error initialising encoder model from %s\n", argv[3]);
            exit(1);       
        }
    } else if (init_radeenc(&enc_model, radeenc_arrays, input_dim) != 0) {
        fprintf(stderr, "Error initialising built-in encoder model\n");
        exit(1);        
    }
    rade_init_encoder(&enc_state);

    assert(enc_model.enc_zdense.nb_outputs == RADE_LATENT_DIM);

    float features_read[frames_per_step*nb_total_features];
    float features[input_dim];
    float z[RADE_LATENT_DIM];
    
    // From celt/cpu_support.h:
    /* We currently support 5 x86 variants:
    * arch[0] -> non-sse
    * arch[1] -> sse
    * arch[2] -> sse2
    * arch[3] -> sse4.1
    * arch[4] -> avx
    */
    int arch = 0;
    
    // This auto-magically selects best arch
    // arch = opus_select_arch();

    fprintf(stderr, "arch: %d bottleneck: %d auxdata: %d input_dim: %d n_z_out: %d\n", 
            arch, bottleneck, auxdata, input_dim, enc_model.enc_zdense.nb_outputs);
    int nb_feature_vecs = 0;
    size_t to_read, nb_read;
    to_read = frames_per_step*nb_total_features;
    while((nb_read = fread(features_read, sizeof(float), to_read, stdin)) == to_read) {
        for (int i=0; i<frames_per_step; i++) {
            for(int j=0; j<num_used_features; j++)
                features[i*num_features+j] = features_read[i*nb_total_features+j];
            if (auxdata)
                features[i*num_features+num_used_features] = -1.0;
        }
        rade_core_encoder(&enc_state, &enc_model, z, features, arch, bottleneck);
        fwrite(z, sizeof(float), RADE_LATENT_DIM, stdout);
        fflush(stdout);
        nb_feature_vecs++;
    }
    fprintf(stderr, "%d feature vectors processed\n", nb_feature_vecs);

    if (argc == 4) {
        munmap(data, len);
        close(fd);
        free(list);
    }

    return 0;
}
