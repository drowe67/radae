# Radio Autoencoder - RADAE

A hybrid Machine Learning/DSP system for sending speech over HF radio channels.  The RADAE encoder takes vocoder features such as pitch, short term spectrum and voicing and generates a sequence of analog PSK symbols.  These are placed on OFDM carriers and sent over the HF radio channel. The decoder takes the received PSK symbols and produces vocoder features that can be sent to a speech decoder.  The system is trained to minimise the distortion of the vocoder features over HF channels.  Compared to classical DSP approaches to digital voice, the key innovation is jointly performing transformation, prediction, quantisation, channel coding, and modulation in the ML network.

## Scope 

This repo is intended to support the authors experimental work, with just enough information for the advanced experimenter to reproduce aspects of the work.  The focus is on waveform development, not software configuration.  It is not intended to be a polished distribution for general use or to work across multiple Linux distros and operating systems - that will come later.  Unless otherwise stated, the code is this repo is intended to run only on Ubuntu Linux 22 on a non-virtual machine.

# Quickstart

1. Installation section
1. Inference section
1. If you would like to transmit/receive test files over HF radios: Over the Air/Over the Cable (OTA/OTC)

# Attributions and License

This software was derived from RDOVAE Python source (Github xiph/opus.git opus-ng branch opus/dnn/torch/rdovae):

J.-M. Valin, J. Büthe, A. Mustafa, [Low-Bitrate Redundancy Coding of Speech Using a Rate-Distortion-Optimized Variational Autoencoder](https://jmvalin.ca/papers/valin_dred.pdf), *Proc. ICASSP*, arXiv:2212.04453, 2023. ([blog post](https://www.amazon.science/blog/neural-encoding-enables-more-efficient-recovery-of-lost-audio-packets))

The RDOVAE derived Python source code is released under the two-clause BSD license.

# Files

| File | Description |
| --- | --- |
| radae/radae.py | ML model and channel simulation |
| radae/dataset.py | loading data for training |
| train.py | trains models |
| inference.py | Testing models, injecting channel impairments, simulate modem sync |
| rx.py | Stand alone receiver, use inference.py as transmitter |
| inference.sh | helper script for inference.py |
| rx.sh | helper script for rx.py |
| ofdm_sync.sh | generates curves to evaluate classical DSP sync performance |
| evaluate.sh | script to compare radae to SSB, generates plots and speech samples |
| evaluate_loop.sh | script to run evaluate.sh over a range of SNRs and channels |
| doppler_spread.m | Octave script to generate Doppler spreading samples |
| load_f32.m | Octave script to load .f32 samples |
| multipath_samples.m | Octave script to generate multipath magnitude sample over a time/freq grid |
| plot_specgram.m | Plots sepctrogram of radae modem signals |
| radae_plots.m | Helper Octave script to generate various plots |
| radio_ae.[tex,pdf] | Latex documenation |
| ota_test.sh | Script to automate Over The Air (OTA) testing |
| Radio Autoencoder Waveform Design.ods | Working for OFDM waveform, including pilot and cyclic prefix overheads |
| compare_models.sh | Builds loss versus Eq/No curves for models to objectively compare |
| est_snr.py | Prototype pilot sequence based SNR estimator - doesn't work for multipath |
| test folder | Helper scripts for ctests |
| loss.py | Tool to calculate mean loss between two feature files, a useful objective measure |
| ml_pilot.py | Training low PAPR pilot sequence |
| stateful_decoder.[py,sh] | Inference test that compares stateful to vanilla decoder |
| stateful_encoder.[py,sh] | Inference test that compares stateful to vanilla encoder |
| radae_tx.[py,sh] | streaming RADAE encoder and helper script |
| radae_rx.[py,sh] | streaming RADAE decoder and helper script |
| resource_est.py | WIP estimate CPU/memory resources |

# Installation

## Packages

sox, python3, python3-matplotlib and python3-tqdm, octave, octave-signal, cmake.  Pytorch should be installed using the instructions from the [pytorch](https://pytorch.org/get-started/locally/) web site. 

## codec2-dev

Supplies some utilities used for `ota_test.sh` and `evaluate.sh`
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

# Automated Tests

The `cmake/ctest` framework is being used as a build and test framework. The command lines in `CmakeLists.txt` are a good source of examples, if you are interested in running the code in this repo. The ctests are a work in progress and may not pass on all systems (see Scope above).

To run the cests:
```
cd radae/build
ctest
```
To list tests `ctest -N`, to run just one test `ctest -R inference_model5`, to run in verbose mode `ctest -V -R inference_model5`.  You can change the paths to `codec2-dev` on the `cmake` command line:
```
cmake -DCODEC2_DEV=~/tmp/codec2-dev ..
```
A lot of the tests generate a float IQ sample file.  You can listen to this file with: 
```
cat rx.f32 | python3 f32toint16.py --real --scale 8192 | play -t .s16 -r 8000 -c 1 - bandpass 300 2000
```
The scaling `--scale` is required as the low SNRs mean the noise peak amplitude can clip 16 bit samples if not carefully scaled.

# Inference

`inference.py` is used for inference, which has been wrapped up in a helper script `inference.sh`.  Inference runs by default on the CPU, but will run on the GPU with the `--cuda-visible-devices 0` option.

1. Generate `out.wav` at the default Eb/No = 100 dB:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav out.wav
   ```

1. Play output sample to your default `aplay` sound device at BPSK Eb/No = 3dB:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav - --EbNodB 3
   ```

1. Vanilla LPCNet-fargan (ie no analog VAE) for comparison:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav - --passthru
   ```

1. Multipath demo at approx 0dB B=3000 Hz SNR. First generate multipath channel samples using GNU Octave (only need to be generated once): 
   ```
   octave:85> Rs=50; Nc=20; multipath_samples("mpp", Rs, Rs, Nc, 60, "h_mpp.f32")
   $ ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth ~/LPCNet/wav/all.wav tmp.wav --EbNodB 3 --write_latent z_hat.f32 --h_file h_mpp.f32
   ```
   Then use Octave to plot scatter diagram using z_hat latents from channel:
   ```
   octave:91> radae_plots; do_plots('z_hat.f32') 
   ```

# Multipath rate Fs

1. Baseline no noise simulation on Multipath Poor MPP channel:
   ```
   octave:85> Fs=8000; Rs=50; Nc=20; multipath_samples("mpp", Fs, Rs, Nc, 60, "h_mpp.f32","g_mpp.f32")
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --write_latent z.f32 --write_rx rx.f32 --pilots --pilot_eq --eq_ls --ber_test --EbNo 100 --g_file g_mpp.f32 --cp 0.004
   octave:87> radae_plots; do_plots('z.f32','rx.f32')
   ```

1. Multipath Disturbed (MPD) demo:
   ```
   ./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav brian_g8sez_mpd_snr3dB.wav --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --EbNodB 6 --g_file g_mpd.f32 --write_rx rx.f32
   ```
   Optional plots (e.g. spectrogram):
   ```
   octave:40> radae_plots; do_plots('z.f32','rx.f32')
   ```
   Listen to "off air" signal.
   ```
   cat rx.f32 | python3 f32toint16.py --real --scale 8192 | sox -t .s16 -r 8000 -c 1 - brian_g8sez_mpd_snr3dB_rx.wav sinc 0.3-2.7k
   ```

# Evaluate script

Automates joint simulation of SSB and RADAE, generates wave files and spectrograms.  Can adjust noise for equal C/No or equal P/No.

1. Run `peter.wav`` through RADAE and SSB using Eb/No=16dB set point, calibrating SSB noise such that P/No is the same for both samples
   ```
   ./evaluate.sh model17/checkpoints/checkpoint_epoch_100.pth wav/peter.wav 240521 16 --bottleneck 3 -d --peak
   ```
1. As above but MPP channel using Eb/No=6dB set point
   ```
   ./evaluate.sh model17/checkpoints/checkpoint_epoch_100.pth wav/peter.wav 240521 6 --bottleneck 3 -d --peak --g_file g_mpp.f32
   ```
1. With dim=40 mixed rate model, and we run Eb/No 3dB higher than a dim=80 model.  `g_nc20_mpp.f32` is a time domain fading file so works for Nc=20 and Nc=10.
   ```
   ./evaluate.sh model18/checkpoints/checkpoint_epoch_100.pth wav/peter.wav 240524 9 --bottleneck 3 -d --peak --latent_dim 40 --g_file g_mpp.f32
   ```

# Simulation of Seperate Tx and Rx

We separate the system into a transmitter `inference.py` and stand alone receiver `rx.py`.  These examples test the OFDM waveform, including pilot symbol insertion, cyclic prefix, least squares phase EQ, and coarse magnitude EQ.

BER tests are useful to calibrate the system, and measure loss from classical DSP based synchronisation.  We have a good theoretical model for the expected BER on AWGN and multipath channels.

1. First generate no-noise reference symnbols for BER measurement `z_100dB.f32`:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots --write_latent z_100dB.f32 --write_rx rx_100dB.f32 --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test
   ```
   `rx_100dB.f32` is the rate Fs IQ sample file, the actual modem signal we could send over the air.

1. The file `z_100dB.f32` can then be used as a reference to measure BER at the receiver, e.g. using the no noise `rx_100dB.f32` sample:
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_100dB.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```

1. An AWGN channel at Eb/No = 0dB, first generate `rx_0dB.f32`:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots --write_rx rx_0dB.f32 --EbNodB 0 --cp 0.004 --pilot_eq --eq_ls --ber_test
   ```
   Then demodulate
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_0dB.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```
   This will give a BER of around 0.094, compared to 0.079 theoretical for BPSK, a loss of about 1dB due to non-ideal synchronisation.

1. Compare to the BER with the ideal phase estimate (pilot based EQ disabled):
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav t.wav --pilots --rate_Fs --EbNodB 0 --ber_test
   ```
   Which is exactly the theoretical 0.078. Note the ideal BER for AWGN is given by `BER = 0.5*erfc(sqrt(Eb/No))`, where Eb/No is the linear Eb/No.

1. Lets introduce frequency, and magnitude (gain) offsets typical of real radio channels:
   ```
   /inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --pilots --write_rx rx_0dB.f32 --EbNodB 0 --cp 0.004 --pilot_eq --eq_ls --ber_test --freq_offset 2 --gain 0.1
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_0dB.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```
   The acquisition system detected and corrected the 2Hz frequency offset, and the -20dB (--gain 0.1) magntitude offset, and the resulting BER was about the same at 0.095.

1. Typical HF multipath channels evolve at around 1 Hz, so it's a good idea to use longer samples to get a meaningful average.  First generate the reference `z_100dB.f32` file for `all.wav`:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --rate_Fs --pilots --write_latent z_100dB.f32 --write_rx rx_100dB.f32 --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test
   ```
   Check that it's doing sensible things with no noise (BER=0):
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_100dB.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```
   Lets generate a MPP sample `rx_100dB_mpp.f32` with no noise, and check the scatter diagram is a nice cross shape with BER close to 0:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --rate_Fs --pilots --write_rx rx_100dB_mpp.f32 --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test --g_file g_mpp.f32
   ```
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_100dB_mpp.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```
   You can listen to the modulated OFDM waveform over the MPP channel simulation with:
   ```
   play -r 8k -e float -b 32 -c 2 rx_100dB_mpp.f32 sinc 0.3-2.7k
   ```
   Lets add some channel impairments:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --rate_Fs --pilots --write_rx rx_0dB_mpp.f32 --EbNodB 0 --cp 0.004 --pilot_eq --eq_ls --ber_test --g_file g_mpp.f32
   ```
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_0dB_mpp.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
   ```
   Which gives us a BER of 0.172, about 1.5dB from the ideal Rayleigh multipath channel BER of 0.15 (which the MPP model approximates).

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
   The output will be a `~/Downloads/kiwisdr_usb_radae.wav` and `~/Downloads/kiwisdr_usb_ssb.wav`, which you can listen to and compare, `~/Downloads/kiwisdr_usb_spec.png` is the spectrogram.  The C/No will be estimated and displayed but this is unreliable at present for non-AWGN channels.  The `ota_test.sh` script is capable of automatically recording from KiwiSDRs, however this code hasn't been debugged yet.

# Streaming

`radae_rx.py` is s streaming receiver that accepts IQ samples on stdin, and outputs z vectors on stdout.  To listen to an example decode:
```
./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --write_rx rx.f32
cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth | build/src/lpcnet_demo -fargan-synthesis - - | aplay -f S16_LE -r 16000
```
To run just the core streaming decoder:
```
cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth > features_rx_out.f32
```
Full RADAE Streaming Rx in real time (fom off air audio samples to speaker):
```
cd ~/radae
cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 1 | ./build/src/lpcnet_demo -fargan-synthesis - - | aplay -f S16_LE -r 16000
```
Ctest that measures % CPU used:
```
cd ~/radae/build
ctest -V -R radae_rx_fargan
<snip>
run time:  6.41 duration:  9.82 percent CPU: 65.26
```

## Profiling example

Total run time for 50 second `all/wav`:
```
ctest -R radae_rx_dfdt
time cat rx.f32 | python3 radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 0 --no_stdout
```
Profiling each function, using shorter wave file:
```
ctest -V -R radae_rx_basic
cat rx.f32 | python3 -m cProfile -s time radae_rx.py model17/checkpoints/checkpoint_epoch_100.pth -v 0 --no_stdout | more
```

# Training

This section is optional - pre-trained models that run on a standard laptop CPU are available for experimenting with RADAE. If you wish to perform training, a serious NVIDIA GPU is required - the author used a RTX4090.

1. Generate a training features file using your speech training database `training_input.pcm`, we used 200 hours of speech from open source databases:
   ```
   ./lpcnet_demo -features training_input.pcm training_features_file.f32
   ```
   
1. Vanilla fixed Eb/No:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 --plot_loss training_features_file.f32 model_dir_name
   ```

1. Rate Rs with multipath, over range of Eb/No:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model05 --mp_file h_mpp.f32 --range_EbNo --plot_loss
   ```

1. Rate Fs with simulated PA:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model06 --plot_loss --rate_Fs --range_EbNo
   ```

1. Rate Fs with phase and freq offsets:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model07 --range_EbNo --plot_loss --rate_Fs --freq_rand
   ```

1. Generate `loss` versus Eb/No curves for a model:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 1 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 tmp --range_EbNo --plot_EqNo model05 --rate_Fs --initial-checkpoint model05/checkpoints/checkpoint_epoch_100.pth
   ```
   This runs another training epoch (results not used so saved in `tmp` folder), but the results won't change much as the network has converged.  While training across a range of Eb/No, it gathers stats on `loss` against Eq/No, and plots them in a PNG and dumps a text file.  The text output is useful for plotting curves from different training runs together. TODO: reconcile original Eb/No simulation parameter with latest model
   that also trains constellations which means symbol energy Eq is a better parameter.

   Octave can be used to plot several loss curves together:
   ```
   octave:120> radae_plots; loss_EqNo_plot("loss_models",'model05_loss_EbNodB.txt','m5_Rs_mp','model07_loss_EbNodB.txt','m7_Fs_offets','model08_loss_EbNodB.txt','m8_Fs')
   ```

1. (May 2024) Training dim=80 mixed rate PAPR optimised model.  Note we need the Nc=20 version of the multipath H matrix `h_nc20_train_mpp.f32` as fading is aplies at rate Rs.  Bottleneck 3 is a tanh() on the magnitude of the complex rate Fs time domain samples.  The SNR ends up about 3dB higher, as discussed in the mixed rate/noise calibration section of the Latext doc: 
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model19 --bottleneck 3 --h_file h_nc20_train_mpp.f32 --range_EbNo --plot_loss
   ```

1. (May 2024) Training dim=40 mixed rate PAPR optimised model.  Note we need the Nc=10 version of the multipath H matrix `h_nc10_train_mpp.f32`.  Bottleneck 3 is a tanh() on the magnitude of the complex rate Fs time domain samples.  We bump the range of Eb/Nos trained over by 3dB `--range_EbNo_start -3` as a 10 carrier waveform will have 3dB more energy per symbol.
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model18 --latent-dim 40 --bottleneck 3 --h_file h_nc10_train_mpp.f32 --range_EbNo_start -3 --range_EbNo --plot_loss
   ```

1, (Aug 2024) Training model with auxillary/embedded data at 25 bits/:
    ```
    python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model19_check3 --bottleneck 3 --h_file h_nc20_train_mpp.f32 --range_EbNo --plot_loss --auxdata
    ```

# Models & samples

A log of models trained by the author.

| Model | Description | Train at | Samples |
| ---- | ---- | ---- | ---- |
| model01 | trained at Eb/No 0 dB | Rs | - |
| model02 | trained at Eb/No 10 dB | Rs | - |
| model03 | --range_EbNo -2 ... 13 dB, modified sqrt loss |Rs | - |
| model04 | --range_EbNo -2 ... 13 dB, orginal loss, noise might be 3dB less after calibration | Rs | - | 
| model05 | --range_EbNo, --mp_file h_mpp.f32, sounds good on MPP and AWGN at a range of SNR - no pops | Rs | 240221_m5_Rs_mp |
| model06 | --range_EbNo, --rate_Fs, trained on AWGN with PA model, PAPR about 1dB, OK at a range of Eb/No | Fs |240223_m6_Fs_papr |
| model07 | --range_EbNo, -6 ... 14, --rate_Fs, AWGN freq, phase, gain offsets, some degredation at 0dB | Fs | 240301_m7_Fs_offets | Fs |
| model08 | --range_EbNo, -6 ... 14, --rate_Fs, AWGN no offsets (vanilla rate Fs), similar to model 05 | Fs | 240301_m8_Fs | 
| model05 | practical OFDM with 4ms CP and pilots, increased Rs', MPP, 4dB sync loss, speech dropping in and out | Fs | 240319_m5_Fs_mp | 
| model05 | practical OFDM with 120ms modem frame, 4ms CP and pilots, reduced Rs', 1dB improvement, mooneer sample | Fs | 240320_m5_Fs | 
| model05 | HF OTA tests of up to 2000km, including weak signal, EMI, NVIS | Fs | 240326_ota_hf | 
| model09 | repeat of model05 as a sanity check, similar loss v Eq/No | Rs | |
| model10 | First attempt at dim=40 1D bottleneck, AWGN, poor audio quality, relatively high loss v Eq/No, square constellation | Rs | |
| model11 | dim=40 with 2D bottleneck #1, AWGN, good audio quality, similar to model05, good loss v Eq/No, circular constellation | Rs | |
| model12 | dim=40 with 2D bottleneck #2, AWGN, a more sensible --range_EbNo_start 0, improved loss v Eq/No, circular constellation | Rs | |
| model13 | dim=80 with 2D bottleneck 3 on rate Fs, AWGN, 0.5db PAPR, loss > m5, a few dB poorer at low SNR, very similar at high SNR | Fs | |
| model14 | dim=80 with 2D bottleneck 3 on rate Fs, 10 hour --h_file h_nc20_test.f32 --range_EbNo_start 0, 0.7dB PAPR, "accident" as it introduces phase distortion with no EQ, but does a reasonable job (however speech quality < m5), handles phase and small freq offsets with no pilots, worth exploring further | Fs | |
| model15 | repeat of model05/09 with 250 hour --h_file h_nc20_train_mpp.f32, after refactoring dataloader, loss v epoch curve v close to model09, ep 100 loss 0.150 | Rs | |
| model16 | repeat of model05/09 with 10 hour --h_file h_nc20_test.f32, testing short h file, ep 100 loss 0.149 | Rs | |
| model17 | `--bottleneck 3 --h_file h_nc20_train_mpp.f32` mixed rate Rs with time domain bottelneck 3, ep 100 loss 0.112 | Rs | 240601_m17 |
| model18 | `--latent-dim 40 --bottleneck 3 --h_file h_nc10_train_mpp.f32 --range_EbNo_start -3` like model17 but dim 40, ep 100 loss 0.123 | Rs | 240601_m18 |
| model05_auxdata | model05 (rate Rs h_nc20_train_mpp.f32) with --auxdata 100 bits/s see PR#13 | Rs | - |
| model05_auxdata25 | model05 (rate Rs h_nc20_train_mpp.f32) with --auxdata 25 bits/s see PR#13 | Rs | - |
| model19 | like model17 but with 25 bits/s auxdata, ep 100 loss 0.124 | Fs | - |
| model19_check3 | model19 but loss function weighting for data symbols redcued fom 1/18 to 0.5/18, which reduced vocoder feature loss with just a small impact on BER.  Loss at various op points and channels very close to model17 | Fs | - |

Note the samples are generated with `evaluate.sh`, which runs inference at rate Fs. even if (e.g model 05), trained at rate Rs.

# Specifications

Using model 17 waveform:

| Parameter | Value | Comment |
| --- | --- | --- |
| Audio Bandwith | 100-8000 Hz | |
| RF Bandwidth | 1500 Hz (-6dB) | |
| Tx Peak Average Power Ratio | < 1dB | |
| Threshold SNR | -3dB | AWGN channel, 3000 Hz noise bandwidth |
| Threshold C/No | 32 dBHz | AWGN channel |
| Threshold SNR | 0dB | MPP channel (1Hz Doppler, 2ms path delay), 3000 Hz noise bandwidth |
| Threshold C/No | 35 dBHz | MPP channel |
| Frame size | 120ms | algorithmic latency |
| Modulation | OFDM | discrete time, continously valued symbols |
| Vocoder | FARGAN | low complexity ML vocoder |
| Total Payload Symbol rate | 2000 Hz | payload data symbols, all carriers combined |
| Number of Carriers | 30 | |
| Per Carrier Symbol rate | 50 Hz | |
| Cyclic Prefix | 4ms | |
| Worst case channel | MPD: 2Hz Doppler spread, 4ms path delay | Two path Watterson model |
| Mean acquisition time | < 1.5s | 0dB SNR MPP channel | 
| Acquisition frequency range | +/- 50 Hz | |
| Acquisition co-channel interference tolerence | -3dBC | Interfering sine wave level, <2s mean acquisition time |
| Auxilary text channel | No | |
| SNR measurement | No | |
| Tx and Rx sample clock offset | 200ppm | e.g. Tx sample clock 8000 Hz, Rx sample clock 8001 Hz |

# Web based Stored File Processing

This section contains some notes on setting up a web server to run `ota_test.sh`.  The idea is to make it easier for non-Linux users to contribute to the stored file test program.  The general idea is a CGI script interfaces to `ota_test.sh` to perform the Tx and Rx processing.  We configure the web server so that the HTML forms and CGI scripts run in `~/public_html`.  The notes below are for Apache on Ubuntu 22. 

1. The Python packages need to be available system wide , so `www-data` can use them: 
   ```
   sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   sudo -u www-data python3 -c "import torch"
   ```

   ```
   sudo pip3 install matplotlib
   sudo -u www-data python3 -c "import matplotlib"
   ```
   The presence of the packages can be checked by mimicing the www-data user (the last line in each step above should not fail if all is well).

1. Configure Apache for CGI and serving pages from our `~/public_html` dir.
   ```
   sudo a2enmod cgid
   sudo a2enmod userdir
   sudo systemctl restart apache2
   ```
   We want html and cgi to run out of ~/public_html, so permissions have to be `755` and `www-data` has to be added to the users group.
   ```
   mkdir ~/public_html
   chmod 755 public_html
   sudo usermod -a -G <username> www-data
   ```
   To let CGI scripts run from ~/public_html I placed this in my `/etc/apache2/apache2.conf`:
   ```
   <Directory "/home/<username>/public_html">
      Options +ExecCGI
      AddHandler cgi-script .cgi
   </Directory>  
   ```   
   Then restart apache as above.

1. Create sym links to HTML/CGI scripts in `radae` repo, this allows the script to be part of the RADAE repo:
   ```
   cd ~/public_html
   ln -s ~/radae/public_html/tx_form.html tx_form.html
   ln -s ~/radae/public_html/tx_process.cgi tx_process.cgi
   ``` 

1. Edit the path to `CODEC2_DEV` in `radae/public_html/tx_process.cgi`:
   ```
   my_env["CODEC2_DEV"] = "/home/<username>/codec2-dev"
   ```

1. Note that files created when the CGI process run (e.g. `/tmp/input.wav`) get put in a sandbox rather than directly in `/tmp`.  This is a systemd security feature.  You can find the files with:
   ```
   sudo find /tmp -name input.wav | xargs sudo ls -ld
   -rw-r--r-- 1 www-data www-data 3918458 Aug 15 15:28 /tmp/systemd-private-2fcf85ad243b4da08d79d2e27e0375af-apache2.service-vDE2Dg/tmp/input.wav
   ```

1. Apache error log, good for viewing `ota_test.sh` progress and spotting any issues:
   ```
   tail -f /var/log/apache2/error.log
   ```

# Real Time PTT

WIP notes

## Real Time decode using KiwiSDR

1. Install pulse audio null module
   ```
   pactl load-module module-null-sink sink_name=vsink
   ```
1. Start your web browser, and open a tab to a KiwiSDR.
1. Open `pavucontrol`, *Playback* tab, send web browser sound to NULL module.  Audio from web browser should go silent.
1. We take the audio from the null device monitor output for the input to the RADAE Rx:
   ```
   parec --device=vsink.monitor --rate=8000 --channels=1 | python3 int16tof32.py --zeropad | python3 radae_rx.py model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --auxdata | ./build/src/lpcnet_demo -fargan-synthesis - - | aplay -f S16_LE -r 16000
   ```
1. Try transmitting a RADAE signal:
   ```
   ./ota_test.sh -t radae_test.raw -d -f 7175
   ```
   Where `radae_test.raw` is a RADAE-only sample (i.e. without the chirp and SSB, copied from a temp file generated by `ota_test.sh -x`). If you can't open the SSB radio playback device to radio try closing `pavucontrol`.
1. Other useful pulse audio commands:
   ```
   pactl list sinks short
   pactl list sources short
   pactl list modules
   ```

## Real Time Tx from mic to SSB radio

Work in progress notes, needs a clean up once this settles down.

1. Install null module as above.  Using Settings redirect system sounds to null so default analog sound output is free.

1. Test headset mic to audio:
   ```
   parec --device=17 --rate=16000 --channels=1 --latency=1024 | pacat --device=9 --rate=16000 --channels=1 --latency=1024
   ```
   However this is unreliable, doesn't always start.

1. Input from headset mic, save to file.
   ```
   arecord --device "plughw:CARD=LX3000,DEV=0" -f S16_LE -c 1 -r 16000 | ./build/src/lpcnet_demo -features - - | python3 radae_tx.py model19_check3/checkpointscheckpoint_epoch_100.pth --auxdata | python3 f32toint16.py --real --scale 8192 > t.raw
   ```

1. Test decode with:
   ```
   cat t.raw | python3 int16tof32.py --zeropad | python3 radae_rx.py model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --auxdata | ./build/src/lpcnet_demo -fargan-synthesis - - | aplay -f S16_LE -r 16000
   ```

1. Real time transmit:
   ```
   arecord --device "plughw:CARD=LX3000,DEV=0" -f S16_LE -c 1 -r 16000 | ./build/src/lpcnet_demo -features - - | python3 radae_tx.py model19_check3/checkpoints/checkpoint_epoch_100.pth --auxdata | python3 f32toint16.py --real --scale 8192 | aplay -f S16_LE --device "plughw:CARD=CODEC,DEV=0
   ```
   I keyed radio manually.  I recorded the transmission on a local SDR, then decoded with:
   ```
   sox ~/Downloads/sdr.ironstonerange.com_2024-08-19T22_03_13Z_7185.00_lsb.wav -t .s16 -r 8000 -c 1 - | python3 int16tof32.py --zeropad | python3 radae_rx.py model19_check3/checkpoints/checkpoint_epoch_100.pth -v 2 --auxdata | ./build/src/lpcnet_demo -fargan-synthesis - - | aplay -f S16_LE -r 16000
   ```
