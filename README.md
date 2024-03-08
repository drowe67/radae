# Attributions and License

This software was derived from RDOVAE Python source (Github xiph/opus.git opus-ng branch opus/dnn/torch/rdovae):

J.-M. Valin, J. Büthe, A. Mustafa, [Low-Bitrate Redundancy Coding of Speech Using a Rate-Distortion-Optimized Variational Autoencoder](https://jmvalin.ca/papers/valin_dred.pdf), *Proc. ICASSP*, arXiv:2212.04453, 2023. ([blog post](https://www.amazon.science/blog/neural-encoding-enables-more-efficient-recovery-of-lost-audio-packets))

The RDOVAE derived Python source code is released under the two-clause BSD license.

# LPCNet setup

```
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

# Training

1. Vanilla fixed Eb/No:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 --plot_loss training_features_file.f32 model_dir_name
   ```

1. Rate Rs with multipath, over range of Eb/No:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 moedel05 --mp_file h_mpp.f32 --range_EbNo --plot_loss
   ```

1. Rate Fs with simulated PA:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model06 --plot_loss --rate_Fs --range_EbNo
   ```

1. Rate Fs with phase and freq offsets:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 model07 --range_EbNo --plot_loss --rate_Fs --freq_rand
   ```

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
   octave:85> Rs=50; Nc=20; multipath_samples("mpp", Rs, Rs, Nc, 60, "h.f32")
   $ ./inference.sh model03/checkpoints/checkpoint_epoch_100.pth ~/LPCNet/wav/all.wav tmp.wav --EbNodB 3 --write_latent z_hat.f32 --mp_file h.f32
   ```
   Then use Octave to plot scatter diagram using z_hat latents from channel:
   ```
   octave:91> radae_plots; do_plots('z_hat.f32') 
   ```

1. Generate `loss` versus Eb/No curves for a model:
   ```
   python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 1 --lr 0.003 --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 tmp --range_EbNo --plot_EbNo model05 --rate_Fs --initial-checkpoint model05/checkpoints/checkpoint_epoch_100.pth
   ```
   This runs another training epoch (results not used so saved in `tmp` folder), but the results won't change much as the network has converged.  While training across a range of Eb/No, it gathers stats on `loss` against Eb/No, and plots them in a PNG and dumps a text file.  The text output is useful for plotting curves from different training runs together. One weakness is it doesn't measure actual Eb/No - Eb may vary as different networks have slightly different symbol magnitudes (TODO improve this).

   Octave can be used to plot several loss curves together:
   ```
   octave:120> radae_plots; loss_EbNo_plot("loss_models",'model05_loss_EbNodB.txt','m5_Rs_mp','model07_loss_EbNodB.txt','m7_Fs_offets','model08_loss_EbNodB.txt','m8_Fs')
   ```
   
# OTA/OTC

We separate the system into a transmitter `inference.py` and stand alone receiver `rx.py`.

BER tests, useful to calibrate system, and measures loss from classical DSP based synchronisation.

1. First enerate no-noise reference symnbols for BER measurement `z_100dB.f32`:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav t.wav --EbNodB 100 --pilots --pilot_eq --rate_Fs --ber_test --write_rx rx_100dB.f32 --write_latent z_100dB.f32
   ```
   
3. The file `z_100dB.f32` can then be used to measure BER at the receiver, e.g. with no noise:
   ```
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_100dB.f32 t.wav --pilots --pilot_eq --ber z_100dB.f32 --plots
   ```

4. Or with noise, frequency, and gain offsets:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav t.wav --pilots --pilot_eq --rate_Fs --EbNodB 0 --freq_offset 2 --write_rx rx_0dB.f32 --ber_test
   ./rx.sh model05/checkpoints/checkpoint_epoch_100.pth rx_0dB.f32 t.wav --pilots --pilot_eq --ber z_100dB.f32 --plots
   ```
   
5. Compare to the BER without pilot based EQ:
   ```
   ./inference.sh model05/checkpoints/checkpoint_epoch_100.pth wav/peter.wav t.wav --pilots --rate_Fs --EbNodB 0 --ber_test
   ```
   Note the ideal BER for AWGN is given by `BER = 0.5*erfc(sqrt(Eb/No))`, where Eb/No is the linear Eb/No.

# Tests

1. BER test to check simulation modem calibration `--ber_test`
2. Fixed multipath channel test `--mp_test`.

# Models

| Model | Description | Train at | Samples |
| ---- | ---- |
| model01 | trained at Eb/No 0 dB | Rs | - |
| model02 | trained at Eb/No 10 dB | Rs | - |
| model03 | --range_EbNo -2 ... 13 dB, modified sqrt loss |Rs | - |
| model04 | --range_EbNo -2 ... 13 dB, orginal loss, noise might be 3dB less after calibration | Rs | - | 
| model05 | --range_EbNo, --mp_file h_mpp.f32, sounds good on MPP and AWGN at a range of SNR - no pops | Rs | 240221_m5_Rs_mp |
| model06 | --range_EbNo, --rate_Fs, trained on AWGN with PA model, PAPR about 1dB, OK at a range of Eb/No | Fs |240223_m6_Fs_papr |
| model07 | --range_EbNo, -6 ... 14, --rate_Fs, AWGN freq, phase, gain offsets, some degredation at 0dB | 240301_m7_Fs_offets | Fs |
| model08 | --range_EbNo, -6 ... 14, --rate_Fs, AWGN no offsets (vanilla rate Fs), similar to model 05 | 240301_m8_Fs | Fs |

Note the samples are generated with `evaluate.sh`, which runs inference at rate Fs. even if (e.g model 05), trained at rate Rs.

# Notes

1. Issue: ~We would like smooth degredation from high SNR to low SNR, rather than training and operating at one SNR.  Currently if trained at 10dB, not as good as model trained at 0dB when tested at 0dB.  Also, if trained at 0dB, quality is reduced when tested on high SNR channel, compared to model trained at high SNR.~  This has been dealt with by training at a range of SNRs.

1. Issue: ~occasional pops in output speech (e.g. model01/03/04, vk5dgr_test 0 and 100dB, model03 all 100dB).  Speaker depedence, e.g. predictor struggling with uneven pitch tracks of some speakers?  Network encountering data it hasn't seen before? Some bug unrealted to training?~ model05 with multipath in loop is pop free

1. Issue: vk5dgr_test.wav sounds poorer than LPCNet/wav/all.wav - same speaker, but former much louder.

1. Test: Multipath with no noise should mean speech is not impaired, as no "symbol errors".

1. Test: co-channel interference, interfering sinusoids, and impulse noise, non-flat passband.

1. Test: level sensitivity, do we need/assume amplitude normalisation?

1. Test: Try vocoder with several speakers and background noise.
   + can hear some whitsles on vk5dgr_test.wav with vanilla fargan (maybe check if I have correct version)

1. Can we include maximum likelyhood detection in rx side of the bottelneck?  E.g. a +2 received means very likely +1 was transmitted, and shouldn't have the same weight in decoding operation as a 0 received.  Probability of received symbol.

1. ~Plot scatter diagram of Tx to see where symbols are being mapped.~

1. ~Reshape pairs of symbols to QPSK, as I think effect of noise will be treated differently in a 2D mapping maybe sqrt(2) better.~

1. ~Reshape into matrix with Nc=number of carriers columns to simulate OFDM.~

1. ~Ability to inject different levels of noise at test time.~

1. ~Using OFDM matrix, simulate symbol by symbol fading channel.  This is a key test.  Need an efficient way to generate fading data, maybe create using Octave for now, an hours worth, or rotate around different channels while training.~

1. ~Confirm SNR calculations, maybe print them, or have SNR3k | Es/No as cmd line options~

1. PAPR optimisation.  If we put a bottleneck on the peak power, the network should optimise for miminal PAPR (maximise RMS power) for a given noise level. Be interesting to observe envelope of waveform as it trains, and the phase of symbols. We might need to include sync symbols.

1.~ Way to write/read bottleneck vectors (channel symbols)~

1. Look at bottleneck vectors, PCA, any correlation?  Ameniable to VQ?  Further dim reduction? VQ would enable comparative test using classical FEC methods.

1. How can we apply interleaving, e.g./ just spread symbol sover a longer modem frame, or let network spread them.

1. Diversity in frequency - classical DSP or with ML in the loop?

1. Sweep different latent dimensions and choose best perf for given SNR.

1. Can we use loss function as an objective measure for comparing different schemes?

1. Naming thoughts (what is it):
   * Neural modem - as network selects constellation, or "neural speech modem"
   * Neural channel coding - as network takes features and encoders them for transmisison of the channel
   * Radio VAE or Radio AE - don't feel this has much to do with variational autoencoders
   * Joint source and channel coding
   