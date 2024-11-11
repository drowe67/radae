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

# Fading channel simulation

HF channel sim (two path Rayleigh) is pretty close to TIA-102.CAAA-E 1.6.33 Faded Channel Simulator. The measured level crossing rate (LCR) seems to meet req (f), for v=60 km/hr, f = 450 MHz, and P=1 when measured over a 10 second sample. We've used Rs=2000 symb/s here, so x-axis of plot is 1 second in time.

![LMR 60](doc/lmr_60.png)

```
octave:39>  multipath_samples("lmr60",8000, 2000, 1, 10, "h_lmr60.f32")
Generating Doppler spreading samples...
fd = 25.000
path_delay_s = 2.0000e-04
Nsecplot = 1
Pav = 1.0366
P = 1
LCR_theory = 23.457
LCR_meas = 24.400
```

# Single Carrier PSK Modem

A single carrier PSK modem "back end" that connects the ML symbols to the radio.  This particular modem is written in Python, and can work with DC coupled and passband BBFM radios. It uses classical DSP, rather than ML.  Unlike the HF RADE waveform which used OFDM, this modem is single carrier.

1. Run a single test with some plots, Eb/No=4dB, 100ppm sample clock offset, BER should be about 0.01:
   ```
   python3 -c "from radae import single_carrier; s=single_carrier(); s.run_test(100,sample_clock_offset_ppm=-100,plots_en=True,EbNodB=4)"
   ```
1. Run a suite of tests:
   ```
   ctest -V -R bbfm_sc
   ```
1. Create a file of BBFM symbols, 80 symbols every 40ms, plays expected output speech:
    ```
    ./bbfm_inference.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav - --write_latent z.f32
    ```
2. Sanity check of modem, BER test using digital, BPSK symbols, the symbols in z.f32 are replaced with BPSK symbols. `t.int16` is a real valued Fs=9600Hz sample file, that could be played into a FM radio.
   ```
   cat z.f32 | python3 sc_tx.py --ber_test > t.int16
   cat t.int16 | python3 sc_rx.py --ber_test --plots > /dev/null
   ```
3. Send the BBFM symbols over the modem, and listen to results:
    ```
   cat z.f32 | python3 sc_tx.py > t.int16
   cat t.int16 | python3 sc_rx.py > z_hat.f32
   ./bbfm_rx.sh model_bbfm_01/checkpoints/checkpoint_epoch_100.pth z_hat.f32 -
    ```
4. Compare MSE of features passed through the system, first with z == z_hat, then with z passed through modem to get z_hat:
   ```
     python3 loss.py features_in.f32 features_out.f32
     loss: 0.033
     python3 loss.py features_in.f32 features_rx_out.f32
     loss: 0.035
   ```
  This is a really good result, and likely inaudible. The `feature*.f32` files are produced as intermediate outputs from the `bbfm_inference.sh` and `bbfm_rx.sh` scripts.

5. Playing samples over a USB sounds card connected to a radio, note selection of sample rate:
   ```
   aplay --device="plughw:CARD=Audio,DEV=0" -r 9600 -f S16_LE t1.int16
   ```

6. Feeding samples from an off air wave file captured from a Rx to demod. Note `sc_xx` tools default to a centre freq of 1500Hz
   ```
   sox ~/Desktop/sc-ber-003.wav -t .s16 -r 9600 -c 1 - highpass 100 | python3 sc_rx.py --plots > z_hat.f32
   ```