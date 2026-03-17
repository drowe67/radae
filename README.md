# RADE: A Neural Codec for Transmitting Speech over HF Radio Channels

This branch contains source code to support the paper: 

D. Rowe, J.-M. Valin, [RADE: A Neural Codec for Transmitting Speech over HF Radio Channels](https://arxiv.org/abs/2505.06671), arXiv:2505.06671, 2025. 

The RADE source code is released under the two-clause BSD license.

For acronyms RADAE and RADE can be used interchangeably.  RADAE was the original name of the project.

# Quickstart

1. Installation section
1. Inference section

# Files

| File | Description |
| --- | --- |
| radae/radae.py | ML model and channel simulation |
| radae/dataset.py | loading data for training |
| radae_base.py | Shared ML code between models |
| train.py | trains models |
| inference.py | Testing models, injecting channel impairments, simulate modem sync |
| rx.py | Stand alone receiver, use inference.py as transmitter |
| inference.sh | helper script for inference.py |
| rx.sh | helper script for rx.py |
| doppler_spread.m | Octave script to generate Doppler spreading samples |
| load_f32.m | Octave script to load .f32 samples |
| multipath_samples.m | Octave script to generate multipath channel simulation samples |
| plot_specgram.m | Plots spectrogram of radae modem signals |
| radae_plots.m | Helper Octave script to generate various plots |
| ota_test.sh | Script to automate Over The Air (OTA) testing |
| Radio Autoencoder Waveform Design.ods | Working for OFDM waveform, including pilot and cyclic prefix overheads |
| compare_models.sh | Builds loss versus Eq/No curves for models to objectively compare |
| test folder | Helper scripts for ctests |
| loss.py | Tool to calculate mean loss between two feature files, a useful objective measure |
| stateful_decoder.[py,sh] | Inference test that compares stateful to vanilla decoder |
| stateful_encoder.[py,sh] | Inference test that compares stateful to vanilla encoder |
| radae_tx.[py,sh] | streaming RADAE encoder and helper script |
| radae_rx.[py,sh] | streaming RADAE decoder and helper script |

# Installation

This code is designed to run on Ubuntu 22.

## Packages

sox, python3, python3-matplotlib and python3-tqdm, octave, octave-signal, cmake.  Pytorch should be installed using the instructions from the [pytorch](https://pytorch.org/get-started/locally/) web site. 

## codec2-dev

Supplies some utilities used for the automated ctests and over the air test scripts
```
cd ~
git clone https://github.com/drowe67/codec2-dev.git
cd codec2-dev
mkdir build_linux
cd build_linux
cmake -DUNITTEST=1 ..
make ch mksine tlininterp
```

## RADAE

Builds the FARGAN vocoder and ctest framework, most of RADAE is in Python.
```
cd ~
git clone https://github.com/drowe67/radae.git
cd radae
mkdir build
cd build
cmake ..
make
```

# Inference

The WASPAA 2025 paper was based on the `model19_check3` model.

1. Take the sample `wav/brian_g8sez.wav` and pass it through the RADE system at the default (very high) SNR, producing the output file `out.wav`:
   ```
   ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav out.wav --auxdata --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16
   aplay wav/brian_g8sez.wav out.wav
   ```

1. Vanilla FARGAN (ie no RADE encoder/decoder) for comparison:
   ```
   ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav fargan_out.wav --auxdata --passthru
   aplay fargan_out.wav
   ```

1. Lets lower the channel SNR. Play received speech to your default `aplay` sound device, this time at around 0dB SNR:
   ```
   ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav - --auxdata --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --EbNodB 3
   ```

1. HF multipath demo at approx 0dB SNR. First generate time domain multipath channel samples using GNU Octave (only need to be generated once): 
   ```
   octave-cli
   octave:85> Fs=8000; Rs=50; Nc=20; multipath_samples('mpp', Fs, Rs, Nc, 3600, '','g_mpp.f32');
   ```
   Then run the simulation using the multipath samples to simulate the channel:
   ```
   ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav out_mpp.wav --auxdata --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --time_offset -16 --EbNodB 3 --g_file g_mpp.f32 --write_latent z_hat.f32 --write_rx rx.f32
   aplay out_mpp.wav
   ```
   You can using Octave to plot a histogram of the `z_hat.f32` vectors (Fig 3), and spectrogram of the received signal `rx.f32` (Fig 7):
   ```
   octave:91> radae_plots; do_plots('z_hat.f32','rx.f32') 
   ```
   You can listen to the simulated "off air" signal (what you would hear on a SSB HF receiver):
   ```
   cat rx.f32 | python3 f32toint16.py --real --scale 8192 | play -t .s16 -r 8000 -c 1 - bandpass 300 2000
   ```

# Automated Tests

The `cmake/ctest` framework is being used as a build and test framework. The command lines in `CMakeLists.txt` are a good source of examples, if you are interested in running the code in this repo.

To run the cests:
```
cd radae/build
ctest
```
To list tests `ctest -N`, to run just one test `ctest -R inference_model5`, to run in verbose mode `ctest -V -R inference_model5`.  You can change the path to `codec2-dev` on the `cmake` command line:
```
cmake -DCODEC2_DEV=~/tmp/codec2-dev ..
```

The scaling `--scale` is required as the low SNRs mean the noise peak amplitude can clip 16 bit samples if not carefully scaled.

# Over the Air/Over the Cable (OTA/OTC)

1. Example of `ota_test.sh` script. `ota_test.sh -x` generates `tx.wav` which contains the simulated SSB and radae modem signals ready to run through a HF radio.  We add noise to create `rx.wav`, then use `ota_test.sh -r` to generate the demodulated audio files `rx_ssb.wav` and `rx_radae.wav`:
   ```
   ./ota_test.sh wav/peter.wav -x 
   ~/codec2-dev/build_linux/src/ch tx.wav - --No -20 | sox -t .s16 -r 8000 -c 1 - rx.wav
   ./ota_test.sh -r rx.wav
   aplay rx_ssb.wav rx_radae.wav
   ```

1. Testing OTA over HF channels. Using my IC7200 as the Tx station:
   ```
   ./ota_test.sh wav/david_vk5dgr.wav -g 6 -t -d -f 14236
   ```
   The `-g 6` is the SSB compressor gain (default 6 so in this case optional); this can be adjusted by experiment, e.g. listening to the `tx.wav` file, and looking for signs of a compressed waveform on Audacity.  To receive the signal I tune into a convenient KiwiSDR, and manually start recording when my radio starts transmitting.  I stop recording when I hear the transmission end.  This will result in a wave file being downloaded.  It's a good idea to trim any excess off the start and end of the rx wave file. It can be decoded with:
   ```
   ./ota_test.sh -d -r ~/Downloads/kiwisdr_usb.wav
   ```
   The output will be a `~/Downloads/kiwisdr_usb_radae.wav` and `~/Downloads/kiwisdr_usb_ssb.wav`, which you can listen to and compare, `~/Downloads/kiwisdr_usb_spec.png` is the spectrogram.

# Training

This section is optional - pre-trained models that run on a standard laptop CPU are available for experimenting with RADAE. If you wish to perform training, a serious NVIDIA GPU is required - the author used a RTX4090.

1. Generate a training features file using your speech training database `training_input.pcm`, we used 200 hours of speech from open source databases:
   ```
   ./lpcnet_demo -features training_input.pcm training_features_file.f32
   ```
   
1. Training model used for WASPAA 2025 paper, first generate 250 hours of frequency domain fading magnitude samples sampled at 50 Hz, then run training (takes around 1 hour):
    ```
    octave:85> Rs=50; Nc=20; multipath_samples("mpp", Rs, Rs, Nc, 250*60*60, "h_nc20_train_mpp.f32")
    python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 training_features_file.f32 model19_check3 --bottleneck 3 --h_file h_nc20_train_mpp.f32 --range_EbNo --plot_loss --auxdata
    ```

# Specifications

Using model19_check3 waveform:

| Parameter | Value | Comment |
| --- | --- | --- |
| Audio Bandwidth | 100-8000 Hz | |
| RF Bandwidth | 1500 Hz (-6dB) | |
| Tx Peak Average Power Ratio | < 1dB | |
| Threshold SNR | -3dB | AWGN channel, 3000 Hz noise bandwidth |
| Threshold C/No | 32 dBHz | AWGN channel |
| Threshold SNR | 0dB | MPP channel (1Hz Doppler, 2ms path delay), 3000 Hz noise bandwidth |
| Threshold C/No | 35 dBHz | MPP channel |
| Frame size | 120ms | algorithmic latency |
| Modulation | OFDM | discrete time, continuously valued symbols |
| Vocoder | FARGAN | low complexity ML vocoder |
| Total Payload Symbol rate | 2000 Hz | payload data symbols, all carriers combined |
| Number of Carriers | 30 | |
| Per Carrier Symbol rate | 50 Hz | |
| Cyclic Prefix | 4ms | |
| Worst case channel | MPD: 2Hz Doppler spread, 4ms path delay | Two path Watterson model |
| Mean acquisition time | < 1.5s | 0dB SNR MPP channel | 
| Acquisition frequency range | +/- 50 Hz | |
| Acquisition co-channel interference tolerance | -3dBC | Interfering sine wave level, <2s mean acquisition time |
| Auxiliary text channel | 25 bits/s | currently used for sync |
| SNR measurement | No | |
| Tx and Rx sample clock offset | 200ppm | e.g. Tx sample clock 8000 Hz, Rx sample clock 8001 Hz |

