# Stored File Test Procedure

Preparing for Tx:
1. Record a wave file of your own voice, see example.  We suggest about 10s but its up to you how long.
1. Try to make the peak level between about half way and the clipping level.
1. Use `ota_test.sh`` to create a `tx.wav file``, this consist of chirp-compressed SSB-radae.
1. Take a look at it on your favourite waveform editor, you can see the signals are adjusted to have the same peak level.

Here is how I test using KiWSDRs:
1. Tune KikWSDR
1. Start transmitting
1. Quickly start KiWiSDR recording (show button)
1. When you hear tranmission stop, stop recording
1. KiwSDR file will be downloaded
1. Load it into your waveform viewer.  I find spectroigram mode useful.  Clip the silence before the chirp, and a little bit of the chirp.  We want the chirp to be the first signal the receiver sees.
1. Add a serial number to the start, and make nyou rown notes of the conditions (e.g. rx station locatioon distance, power level, anything else you think is relevant)
1. Process `ota-rx.sh``
1. listen to the results
1. Try to collect som intersting results, for example different channels nd power levels

TODO:
* Link to April 2024 tests.  Intro, why stored files, thanks for helping.
* Link to installing Software
* How to Test Test using local files
* Note on adjusting SSB compression level with -g option
* Screen shots of files, and listen to tx file
* Ubuntu default sound card when radio plugged in
* Command line examples from a real run
* Further work: automagiclly find chirp
* Description of fields when decoder runs
