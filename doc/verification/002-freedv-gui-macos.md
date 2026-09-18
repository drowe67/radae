# RADE Integration Verification Report

## Application Under Test

| Field | Value |
|---|---|
| Application name | FreeDV |
| Application version / git hash | 3.0.0-dev (git 69473b0) |
| Platform (OS + version) | macOS 26.6.2 (25G83) arm64 (loopback)<br/>macOS 26.6.2 (25G83) x86\_64 (OTAC) |
| Tester name / callsign | Mooneer Salem / K6AQ |
| Date | 2026-09-16 |
| radae repo commit hash | 758e825182f3a216922fb4bbd7f9a4f5c6fe4288 |
| rade\_c repo commit hash | 7eda42f0f10ec5df6ff409527f5855d5b3ed1c94 |

## Signal Path Declaration

- [x] No additional signal processing (AGC, noise gate, resampler, EQ,
      compression) between WAV file input and RADE encoder input during
      this verification test
      <!-- Asserted automatically: test/freedv-ctest-loss.conf.tmpl disables
           AGC, the mic/speaker EQ and Speex noise suppression, and the RADE
           path runs at a fixed 8/16 kHz with no resampling. -->

## Baseline Loss (Step 1)

Re-run with the current repository version before filling in this section.

| Field | Value |
|---|---|
| Baseline loss | 0.079 |
| 10% tolerance window | baseline ± 0.0079  (0.0711 – 0.0869) |

Command used:
```
rade_tx_wav --v2 -f baseline_txfeatures.f32 all.wav baseline_tx.wav
rade_rx_wav --v2 -f baseline_rxfeatures.f32 baseline_tx.wav baseline_decoded.wav
python3 loss.py baseline_txfeatures.f32 baseline_rxfeatures.f32 --clip_start 100 --clip_end 300
```

_Supplied via RADE_LOSS_THRESHOLD._

## Level 1 — Software Loopback (mandatory)

- [x] Pass (loss within ±10% of baseline)

| Field | Value |
|---|---|
| Loss result | 0.083  (PASS) |

Command used / reproduction notes:
```
# Result parsed from an existing run of the command below.
# From the FreeDV build directory (see .github/workflows/cmake-linux.yml /
# .github/workflows/cmake-macos.yml for the per-platform virtual audio setup):
FREEDV_BINARY=<freedv binary> \
FREEDV_RADIO_TO_COMPUTER_DEVICE=<device> \
FREEDV_COMPUTER_TO_RADIO_DEVICE=<device> \
FREEDV_MICROPHONE_TO_COMPUTER_DEVICE=<device> \
FREEDV_COMPUTER_TO_SPEAKER_DEVICE=<device> \
RADE_LOSS_THRESHOLD=0.0869 test/test_rade_loss.sh

# loss.py invocation performed inside test_rade_loss.sh:
python3 rade_src/loss.py txfeatures.f32 rxfeatures.f32 \
    --loss_test <baseline x 1.10> --clip_start 100 --clip_end 300
```

## Level 2 — OTAC: Over The Audio Cable (mandatory for hardware integrations)

- [x] Pass (loss within ±10% of baseline)
- [ ] N/A (software-only integration)

| Field | Value |
|---|---|
| Loss result | 0.083 |
| Sound card (Tx) | Generalplus USB Audio Device (USB VID 0x1B3F, PID 0x2008) + USB isolator |
| Sound card (Rx) | Generalplus USB Audio Device (USB VID 0x1B3F, PID 0x2008) + USB isolator |
| Cable description | ~1m 3.5mm audio cable |

Photo of test setup:
![](002-freedv-gui-macos.jpg)

Reproduction notes:
```
Use Audio MIDI Setup to set both output channels and the single input channel to 0.5.

Command line:

FREEDV_COMPUTER_TO_RADIO_DEVICE="USB Audio Device" FREEDV_RADIO_TO_COMPUTER_DEVICE="USB Audio Device" FREEDV_MICROPHONE_TO_COMPUTER_DEVICE="MacBook Pro Microphone" FREEDV_COMPUTER_TO_SPEAKER_DEVICE="MacBook Pro Speakers" FREEDV_BINARY=/Applications/FreeDV.app/Contents/MacOS/FreeDV ctest -V -R rade_loss
```

## Level 3 — OTC: Over The Coax (optional)

- [ ] Pass (loss within ±10% of baseline)
- [x] Not performed

| Field | Value |
|---|---|
| Loss result | |
| Tx radio | |
| Rx radio | |
| Attenuator(s) | |

Photo of test setup:
_(attach or link)_

Reproduction notes:
```
(paste notes here)
```

## Summary

| Level | Result |
|---|---|
| Level 1 — Software loopback | PASS |
| Level 2 — OTAC | PASS |
| Level 3 — OTC | N/A |

Additional notes:
