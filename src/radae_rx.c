#include <assert.h>
#include <stdio.h>
#ifdef _WIN32
// For _setmode().
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

#include "rade_api.h"

int main(void)
{
    struct rade *r = rade_open("dummy");
    assert(r != NULL);
    int n_features_out = rade_n_features_in_out(r);
    float features_out[n_features_out];
    int n_rx_in = rade_nin_max(r);
    RADE_COMP rx_in[n_rx_in];
    int nin = rade_nin(r);

#ifdef _WIN32
    // Note: freopen() returns NULL if filename is NULL, so
    // we have to use setmode() to make it a binary stream instead.
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

    while((size_t)nin == fread(rx_in, sizeof(RADE_COMP), nin, stdin)) {
        int n_out = rade_rx(r,features_out,rx_in);
        if (n_out) {
            fwrite(features_out, sizeof(float), n_features_out, stdout);
            fflush(stdout);
        }
        nin = rade_nin(r);
    }

    rade_close(r);
    return 0;
}
