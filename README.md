# Radio Autoencoder - RADAE

A hybrid Machine Learning/DSP system for sending speech over HF radio channels.  The RADAE encoder takes vocoder features such as pitch, short term spectrum and voicing and generates a sequence of analog PSK symbols.  These are placed on OFDM carriers and sent over the HF radio channel. The decoder takes the received PSK symbols and produces vocoder features that can be sent to a speech decoder.  The system is trained to minimise the distortion of the vocoder features over HF channels.  Compared to classical DSP approaches to digital voice, the key innovation is jointly performing transformation, prediction, quantisation, channel coding, and modulation in the ML network.

# Quickstart

To Tx and Rx RADAE signal over the air using stored wavefiles you just need these sections:

1. Installation
2. Over the Air/Over the Cable (OTA/OTC)

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
| doppler_spread.m | Octave script to generate Doppler spreading samples |
| load_f32.m | Octave script to load .f32 samples |
| multipath_samples.m | Octave script to generate multipath magnitude sample over a time/freq grid |
| plot_specgram.m | Plots sepctrogram of radae modem signals |
| radae_plots.m | Helper Octave script to generate various plots |
| radio_ae.tex/pdf | Latex documenation |
| ota_test.sh | Script to automate Over The Air (OTA) testing |
| Radio Autoencoder Waveform Design.ods | Working for OFDM waveform, inclduing pilot and cyclic prefix overheads |
| compare_models.sh | Builds loss versus Eq/No curves for models to objectively compare |

# Installation

## Packages

sox, python3, python3-matplotlib and python3-tqdm, octave, octave-signal.  Pytorch should be installed using the instructions from the [pytorch](https://pytorch.org/get-started/locally/) web site. 

## LPCNet setup

```
cd ~
git clone git@github.com:xiph/opus.git
cd opus
git checkout opus-ng
./autogen.sh
./configure --enable-dred
make
cd dnn
```
Initial test:
```
./lpcnet_demo -features input.pcm features.f32
./lpcnet_demo -fargan-synthesis features.f32 output.pcm
```
Playing on a remote machine:
```
scp deep.lan:opus/output.s16 /dev/stdout | aplay -f S16_LE -r 1600
```

## codec2-dev

Supplies some utilities used for `ota_test.sh` and `evaluate.sh`
```
cd ~
git clone git@github.com:drowe67/codec2-dev.gitcd codec2
mkdir build_linux
cd build_linux
cmake -DUNITTEST=1 ..
make ch mksine tlininterp
```
(optional if using HackRF) manually compile misc/tsrc 

# Inference

`inference.py` is used for inference, which has been wrapped up in a helper script `inference.sh`.  Inference runs by default on the CPU, but will run on the GPU with the `--cuda-visible-devices 0` option.

1. Generate `out.wav` at Eb/No = 10 dB:
   ```
   ./inference.sh model01/checkpoints/checkpoint_epoch_100.pth wav/all.wav out.wav --EbNodB 10
   ```

1. Play output sample to your default `aplay` sound device at BPSK Eb/No = 3dB:
   ```
   ./inference.sh model01/checkpoints/checkpoint_epoch_100.pth wav/vk5dgr_test.wav - --EbNodB 3
   ```

1. Vanilla LPCNet-fargan (ie no analog VAE) for comparison:
   ```
   ./inference.sh model01/checkpoints/checkpoint_epoch_100.pth wav/vk5dgr_test.wav - --passthru
   ```

1. Multipath demo at approx 0dB B=3000 Hz SNR. First generate multipath channel samples using GNU Octave (only need to be generated once): 
   ```
   octave:85> Rs=50; Nc=20; multipath_samples("mpp", Rs, Rs, Nc, 60, "h_mpp.f32")
   $ ./inference.sh model03/checkpoints/checkpoint_epoch_100.pth ~/LPCNet/wav/all.wav tmp.wav --EbNodB 3 --write_latent z_hat.f32 --h_file h_mpp.f32
   ```
   Then use Octave to plot scatter diagram using z_hat latents from channel:
   ```
   octave:91> radae_plots; do_plots('z_hat.f32') 
   ```

# Multipath rate Fs

1. Baseline no noise simulation:
   ```
   octave:85> Fs=8000; Rs=50; Nc=20; multipath_samples("mpp", Fs, Rs, Nc, 60, "h_mpp.f32","g_mpp.f32")
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav /dev/null --rate_Fs --write_latent z.f32 --write_rx rx.f32 --pilots --pilot_eq --eq_ls --ber_test --EbNo 100 --g_file g_mpp.f32 --cp 0.004
   octave:87> radae_plots; do_plots('z.f32','rx.f32')
   ```
   
# Seperate Tx and Rx

We separate the system into a transmitter `inference.py` and stand alone receiver `rx.py`.  Theses examples test the OFDM waveform, including pilot symbol insertion, cyclic prefix, least squares phase EQ, and coarse magnitude EQ.

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

1  An AWGN channel at Eb/No = 0dB, first generate `rx_0dB.f32`:
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
   /inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/all.wav /dev/null --rate_Fs --pilots --write_latent z_100dB.f32 --write_rx rx_100dB.f32 --EbNodB 100 --cp 0.004 --pilot_eq --eq_ls --ber_test
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
   ```./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_0dB_mpp.f32 /dev/null --pilots --pilot_eq --cp 0.004 --plots --time_offset -16 --coarse_mag --ber_test z_100dB.f32
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
   ./ota_test.sh wav/david.wav -g 9 -t -d -f 14236
   ```
   The `-g 9` sample gives the `david.wav` sample a little more compression, this was ajusted by experiment, listening to the `tx.wav` file, and looking for signs of a compressed waveform on Audacity.  To receive the signal I tune into a convenient KiwiSDR, and manually start recording when my radio starts transmitting.  I stop recording when I hear the transmission end.  This will result in a wave file being downloaded.  It's a good idea to trim any excess off the start and end of the rx wave file. It can be decoded with:
   ```
   ./ota_test.sh -d -r ~/Downloads/kiwisdr_usb.wav
   ```
   The output will be a `~/Downloads/kiwisdr_usb_radae.wav` and `~/Downloads/kiwisdr_usb_ssb.wav`, which you can listen to and compare, `~/Downloads/kiwisdr_usb_spec.png` is the spectrogram.  The C/No will be estimated and displayed but this is unreliable at present for non-AWGN channels.  The `ota_test.sh` script is capable of automatically recording from KiwiSDRs, however this code hasn't been debugged yet.

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

# Tests

1. BER test to check simulation modem calibration `--ber_test`
2. Fixed multipath channel test `--mp_test`.

# Models & samples

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
| model17 | `--bottleneck 3 --h_file h_nc20_train_mpp.f32` mixed rate Rs with time domain bottelneck 3, ep 100 loss 0.112 | Rs | |
| model18 | `--latent-dim 40 --bottleneck 3 --h_file h_nc10_train_mpp.f32 --range_EbNo_start -3` like model17 but dim 40, ep 100 loss 0.123 | Rs | |

Note the samples are generated with `evaluate.sh`, which runs inference at rate Fs. even if (e.g model 05), trained at rate Rs.

# Notes

1. Issue: vk5dgr_test.wav sounds poorer than LPCNet/wav/all.wav - same speaker, but former much louder.

1. Test: Multipath with no noise should mean speech is not impaired, as no "symbol errors".

1. Test: co-channel interference, interfering sinusoids, and impulse noise, non-flat passband.

1. Test: Try vocoder with several speakers and background noise.
   + can hear some whitsles on vk5dgr_test.wav with vanilla fargan (maybe check if I have correct version)

1. Can we include maximum likelyhood detection in rx side of the bottelneck?  E.g. a +2 received means very likely +1 was transmitted, and shouldn't have the same weight in decoding operation as a 0 received.  Probability of received symbol.

1. Look at bottleneck vectors, PCA, any correlation?  Ameniable to VQ?  Further dim reduction? VQ would enable comparative test using classical FEC methods.

1. How can we apply interleaving, e.g./ just spread symbol sover a longer modem frame, or let network spread them.

1. Diversity in frequency - classical DSP or with ML in the loop?

1. Sweep different latent dimensions and choose best perf for given SNR.

1. Naming thoughts (what is it):
   * Neural modem - as network selects constellation, or "neural speech modem"
   * Neural channel coding - as network takes features and encoders them for transmisison of the channel
   * Radio VAE or Radio AE - don't feel this has much to do with variational autoencoders
   * Joint source and channel coding
   
1. Issue: ~We would like smooth degredation from high SNR to low SNR, rather than training and operating at one SNR.  Currently if trained at 10dB, not as good as model trained at 0dB when tested at 0dB.  Also, if trained at 0dB, quality is reduced when tested on high SNR channel, compared to model trained at high SNR.~  This has been dealt with by training at a range of SNRs.

1. Issue: ~occasional pops in output speech (e.g. model01/03/04, vk5dgr_test 0 and 100dB, model03 all 100dB).  Speaker depedence, e.g. predictor struggling with uneven pitch tracks of some speakers?  Network encountering data it hasn't seen before? Some bug unrealted to training?~ model05 with multipath in loop is pop free

1. ~Test: level sensitivity, do we need/assume amplitude normalisation? ~ Yes - have used pilots to normalise amplitude

1. ~Plot scatter diagram of Tx to see where symbols are being mapped.~

1. ~Reshape pairs of symbols to QPSK, as I think effect of noise will be treated differently in a 2D mapping maybe sqrt(2) better.~

1. ~Reshape into matrix with Nc=number of carriers columns to simulate OFDM.~

1. ~Ability to inject different levels of noise at test time.~

1. ~Using OFDM matrix, simulate symbol by symbol fading channel.  This is a key test.  Need an efficient way to generate fading data, maybe create using Octave for now, an hours worth, or rotate around different channels while training.~

1. ~Confirm SNR calculations, maybe print them, or have SNR3k | Es/No as cmd line options~

1. ~PAPR optimisation.  If we put a bottleneck on the peak power, the network should optimise for miminal PAPR (maximise RMS power) for a given noise level. Be interesting to observe envelope of waveform as it trains, and the phase of symbols. We might need to include sync symbols.~

1.  Way to write/read bottleneck vectors (channel symbols)~

1. ~Can we use loss function as an objective measure for comparing different schemes?~

