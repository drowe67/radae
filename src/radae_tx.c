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
    int n_features_in = rade_n_features_in_out(r);
    float features_in[n_features_in];
    int n_tx_out = rade_n_tx_out(r);
    RADE_COMP tx_out[n_tx_out];

#ifdef _WIN32
    // Note: freopen() returns NULL if filename is NULL, so
    // we have to use setmode() to make it a binary stream instead.
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

    while((size_t)n_features_in == fread(features_in, sizeof(float), n_features_in, stdin)) {
        rade_tx(r,tx_out,features_in);
        fwrite(tx_out, sizeof(RADE_COMP), n_tx_out, stdout);
        fflush(stdout);
    }

    rade_close(r);
    return 0;
}