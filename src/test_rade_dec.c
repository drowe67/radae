/* RADE core decoder test program, z_hat.f32 on stdin, featutres_out.f32 on stdout */

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
#include "rade_dec.h"
#include "rade_dec_data.h"

int opus_select_arch(void);

int main(int argc, char *argv[])
{
    RADEDec      dec_model;
    RADEDecState dec_state;

    if (argc < 2) {
        fprintf(stderr, "usage: %s auxdata[0-1] [weights_blob.bin]\n", argv[0]);
        exit(1);
    }

    int auxdata = atoi(argv[1]);
    int nb_total_features = 36;
    int num_features = 20;
    int num_used_features = 20;
    int frames_per_step = RADE_FRAMES_PER_STEP;
    
    if (auxdata) {
        num_features += 1;
    }
    int output_dim = num_features*frames_per_step;

    int fd;
    void *data;
    int len;
    int nb_arrays;
    struct stat st;
    WeightArray *list;

    if (argc == 3) {
        const char *filename = argv[2];
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
        if (init_radedec(&dec_model, list, output_dim) != 0) {
            fprintf(stderr, "Error initialising decoder model from %s\n", argv[2]);
            exit(1);       
        }
    } else if (init_radedec(&dec_model, radedec_arrays, output_dim) != 0) {
        fprintf(stderr, "Error initialising built-in decoder model\n");
        exit(1);        
    }
    rade_init_decoder(&dec_state);

    assert(dec_model.dec_dense1.nb_inputs == RADE_LATENT_DIM);

    float features_write[frames_per_step*nb_total_features];
    float features[output_dim];
    float z_hat[RADE_LATENT_DIM];
    
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

    fprintf(stderr, "arch: %d auxdata: %d output_dim: %d n_z_in: %d\n", 
            arch, auxdata, output_dim, dec_model.dec_dense1.nb_inputs);
    int nb_latent_vecs = 0;
    size_t to_write = frames_per_step*nb_total_features;
    for(int i=0; i<frames_per_step*nb_total_features; i++) features_write[i] = 0.0;
    
    while(fread(z_hat, sizeof(float), RADE_LATENT_DIM, stdin) == RADE_LATENT_DIM) {
        rade_core_decoder(&dec_state, &dec_model, features, z_hat, arch);
        for (int i=0; i<frames_per_step; i++) {
            for(int j=0; j<num_used_features; j++)
                features_write[i*nb_total_features+j] = features[i*num_features+j];
            if (auxdata)
                features_write[i*nb_total_features+num_used_features] = features[i*num_features+num_used_features];
        }
        fwrite(features_write, sizeof(float), to_write, stdout);
        fflush(stdout);
        nb_latent_vecs++;
    }
    fprintf(stderr, "%d latent vectors processed\n", nb_latent_vecs);

    if (argc == 3) {
        munmap(data, len);
        close(fd);
        free(list);
    }

    return 0;
}
