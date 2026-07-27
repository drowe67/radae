# RADE Integration Verification Report

## Application Under Test

| Field | Value |
|---|---|
| Application name | freedv-ka9q |
| Application version / git hash | [44f80b3](https://github.com/tmiw/freedv-integrations/pull/15/commits/44f80b3dcbfaae9b0f2ceb8459e1f73cbee1f307) |
| Platform (OS + version) | Linux (Fedora 44) - kernel 7.0.12-201.fc44.x86_64 |
| Tester name / callsign | @tmiw |
| Date | 2026-07-27 |
| radae repo commit hash | 25d58fe |
| rade_c repo commit hash | a36161b |

## Signal Path Declaration

- [x] No additional signal processing (AGC, noise gate, resampler, EQ,
      compression) between WAV file input and RADE encoder input during
      this verification test

## Baseline Loss (Step 1)

Re-run with the current repository version before filling in this section.

| Field | Value |
|---|---|
| Baseline loss | 0.081 |
| 10% tolerance window | baseline ± 0.0081 |

Command used:

```
mooneer@fedora:~/radae$ ./build/src/lpcnet_demo -features wav/all.wav features_in.f32
mooneer@fedora:~/radae$ python3 tx2.py 250725/checkpoints/checkpoint_epoch_200.pth features_in.f32 tx.f32
...
mooneer@fedora:~/radae$ python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync tx.f32 features_rx.f32 --quiet
...
mooneer@fedora:~/radae$ python3 loss.py features_in.f32 features_rx.f32 --clip_start 100 --clip_end 300
Loss between features_in.f32 and features_rx.f32
  loss: 0.081 start: 224 acq_time:  1.24 s
```

## Level 1 — Software Loopback (mandatory)

- [x] Pass (loss within ±10% of baseline)

| Field | Value |
|---|---|
| Loss result | 0.083 |

Command used / reproduction notes:

```
mooneer@fedora:~/radae$ python3 f32toint16.py --real --scale 16384 < tx.f32 | sox -t s16 -r 8000 -c 1 - tx_real.wav
mooneer@fedora:~/radae$ cd ~/freedv-integrations/build
mooneer@fedora:~/freedv-integrations/build$ sox ~/radae/tx_real.wav -t s16 -r 8000 -c 1 - | ./src/ka9q/freedv-ka9q --rx-features rxfeatures.f32 >/dev/null
mooneer@fedora:~/freedv-integrations/build$ cd ~/radae
mooneer@fedora:~/radae$ python3 loss.py features_in.f32 ~/freedv-integrations/build/rxfeatures.f32 --clip_start 100 --clip_end 300
```

(Note: freedv-ka9q is RX only.)

## Level 2 — OTAC: Over The Audio Cable (mandatory for hardware integrations)

- [ ] Pass (loss within ±10% of baseline)
- [ ] N/A (software-only integration)

| Field | Value |
|---|---|
| Loss result | 0.091 |
| Sound card (Tx) | GeneralPlus USB Audio Device |
| Sound card (Rx) | GeneralPlus USB Audio Device |
| Cable description | 3.5mm audio cable, approx. 1m length |

Photo of test setup:
![](./freedv-ka9q-oatc-setup.jpg)

Reproduction notes:

```
# Use wpctl status to get correct target IDs for pw-record/pw-play

mooneer@fedora:~/freedv-integrations/build$ pw-record --target 75 --rate 8000 --channels 1 - | ./src/ka9q/freedv-ka9q --rx-features rxfeatures.f32 >/dev/null &
mooneer@fedora:~/freedv-integrations/build$ pw-play --target 73 ~/radae/tx_real.wav

# Needed to flush rest of audio through pipewire
mooneer@fedora:~/freedv-integrations/build$ sox -n silence.wav trim 0 10
mooneer@fedora:~/freedv-integrations/build$ pw-play --target 73 silence.wav 

# Needed to write out feature file and exit freedv-ka9q
mooneer@fedora:~/freedv-integrations/build$ killall pw-record
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
| Level 2 — OTAC | FAIL |
| Level 3 — OTC | N/A |

Additional notes:
