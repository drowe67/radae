# Stored File Test Procedure

This document is a test procedure for the August 2024 RADAE stored file test campaign.  The general idea is to take a 10 second sample of input speech, then send it over a HF radio channel as compressed SSB and RADAE.  This allows a side by side comparsion using the same speech, over approximately the same channel conditions.  Using a stored file makes it possible to repeat the experiment in a controlled fashion over several trials, for example with varying power levels of different receiver locations. As background a similar campaign was conducted in [April 2024](https://freedv.org/?p=595).

Preparing a file for Tx:
1. Record a wave file of your own voice, for example, "Hello, this is VK5XYZ testing the radio Autoencoder 1 2 3 4".
1. The wave file format required is 1 channel, 16 kHz, 16 bit signed integer.
1. We suggest about 10 sec long but feel free to experiment.  The length is not crtical.
1. Try to make the peak level between about half way and the clipping level.
   ![Peak level example](stored_file_level.png)
1. Use a headset microphone and try to avoid room echo and background noise. Don't use a laptop microphone.
1. As examples, there are samples from other hams in `radae/wav`
1. Use `ota_test.sh` to create a `tx.wav`, this consist of chirp-compressed SSB-radae:
   ```
   ota_test.sh -x vk5xyz.wav
   ```
1. You can listen to and plot `tx.wav` with your favourite waveform editor, you can see the signals are adjusted to have the same peak level. The SSB compression gain can be adjusted using the `-g` option; `-g 6` is the default.  Trying going up or down 3dB.  A quieter sample may benefit from more compression.
   ![Peak level example](stored_file_tx.png)

Transmitting your sample:
1. Configure your SSB radio with voice compressor off.  The Tx audio path must be "clean" with no additional processing.
1. The sound card levels should be adjusted to "just move the ALC".
1. Tune the remote receiver (I use a KiwiSDR) to your SSB radio frequency.  You need to be within +/- 50 Hz for the RADAE receiver to acquire.
1. Start transmitting. You can do this manually by playing tx.wav through your transmitter, or use `ota_test.sh`
   ```
   ./ota_test.sh wav/david_vk5dgr.wav -d -f 14236
   ```
   You can adjust the hamlib rig and serial port with command line options, to get help: `ota_test.sh -h`
1. After you start transmitting quickly start the KiwiSDR recording.
1. When you hear transmission stop on the KiwiSDR, stop recording.
1. The KiwiSDR file will be downloaded.
1. Place a serial number in front of the downloaded file to easily identify it e.g. `14_`. Make your own notes of the conditions for that sample (e.g. rx station location, distance, power level, anything else you think is relevant)
1. It's useful to load the file into your waveform viewer.  I find spectrogram mode useful.
   ![Peak level example](stored_file_rx.png)
1. The RADAE receiver will search for the location of the chirp in the first 10 seconds of the sample.  Make sure there is no more than 6 seconds of noise before the chirp starts.  If necessary, edit the file by removing any excess before the chirp starts.
1. If there is more than a few seconds after the RADAE signal stops, clip that off the sample too.
1. Process the receeved sample:
   ```
   ./ota_test.sh -r ~/Downloads/14_sdr.ironstonerange.com_2024-08-08T05_10_14Z_7175.00_lsb.wav
   ```
1. This will locate the chirp, print the C/No and SNR in dB, and generate several other files in the same directory, `_ssb.wav`, `_radae.wav` and a spectrogram `_spec.jpg`.
1. Note the C/No, SNR, and listen to the results, comparing SSB to RADAE.
1. Try to collect some interesting results, for example different channels and power levels, cases where RADAE fails to acquire, intercontinental DX, fast and slow fading, co-channel interference, interference from carriers.

TODO:
* Link to installing Software
* Ubuntu default sound card when radio plugged in
* Description of fields when decoder runs
