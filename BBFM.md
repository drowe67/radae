# Radio Autoencoder - Baseband FM (BBFM)

A version of the Radio Autoencoder (RADE) designed for the baseband FM channel provided by DC coupled and passband FM radios, e.g. land mobile radio (LMR) VHF/UHF use case.

# BBFM ML encoder/decoder

1. First pass training command line:
    ```
    python3 ./train_bbfm.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 --plot_loss ~/Downloads/tts_speech_16k_speexdsp.f32 model_bbfm_01 --range_EbNo --range_EbNo_start 6 --plot_loss
    ```

1. Inference (runs encoder and decoder, and outputs symbols `z_hat.f32`):
    ```
    ./inference_bbfm.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav - --write_latent z_hat.f32
    ```
1. Stand alone decoder, outputs speech from `z_hat.f32` to sound card:
    ```
    ./rx_bbfm.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 -
    ```
1. Or save speech out to a wave file:
    ```
    ./rx_bbfm.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 t.wav
    ```

1. Plot sequence of received symbols:
    ```
    octave:4> radae_plots; do_plots_bbfm('z_hat.f32')
    ```

# Single Carrier PSK Modem

A single carrier PSK modem "back end" that connects the ML symbols to the radio.  This particular modem is written in Python, and can work with DC coupled and passband BBFM radios. It uses classical DSP, rather than ML.  Unlike the HF RADE waveform which used OFDM, this modem is single carrier.

1. Run a single test with some plots, Eb/No=4dB, 100ppm sample clock offset, BER should be about 0.01:
   ```
   python3 -c "from radae import single_carrier; s=single_carrier(); s.run_test(100,sample_clock_offset_ppm=-100,plots_en=True,EbNodB=4)"
   ```
1. Run a suite of tests:
   ```
   python3 -c "from radae import single_carrier; s=single_carrier(); s.run_tests()"
   ```

