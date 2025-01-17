#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
// For _setmode().
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

#include "rade_api.h"

int main(int argc, char *argv[])
{
    rade_initialize();
    int flags = RADE_USE_C_DECODER | RADE_VERBOSE_0;
    /* special test mode that induces a frequency offset error to test UW false sync detection */
    if (argc == 2) {
        if (atoi(argv[1]) == 1) {
            fprintf(stderr, "foff_test\n");
            flags |= RADE_FOFF_TEST;
        }
    }
    struct rade *r = rade_open("dummy", flags);
    assert(r != NULL);
    int n_features_out = rade_n_features_in_out(r);
    float features_out[n_features_out];
    int n_rx_in = rade_nin_max(r);
    RADE_COMP rx_in[n_rx_in];
    int nin = rade_nin(r);
    int n_eoo_bits = rade_n_eoo_bits(r);
    FILE *feoo = fopen("eoo_rx.f32","wb"); assert(feoo != NULL);
    int has_eoo_out;
    float eoo_out[n_eoo_bits];
   
#ifdef _WIN32
    // Note: freopen() returns NULL if filename is NULL, so
    // we have to use setmode() to make it a binary stream instead.
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

    while((size_t)nin == fread(rx_in, sizeof(RADE_COMP), nin, stdin)) {
        int n_out = rade_rx(r,features_out,&has_eoo_out,eoo_out,rx_in);
        if (n_out) {
            fwrite(features_out, sizeof(float), n_features_out, stdout);
            fflush(stdout);
        }
        if (has_eoo_out) {
            fwrite(eoo_out, sizeof(float), n_eoo_bits, feoo);
        }
        nin = rade_nin(r);
        fprintf(stderr, "SNR3k (dB): %d\n", rade_snrdB_3k_est(r));
    }

    rade_close(r);
    rade_finalize();
    fclose(feoo);
    return 0;
}
