# RADE Integration Verification Procedure

## Purpose

This procedure verifies that RADE is correctly integrated into an application, SDR
or hardware radio. The goal is to confirm the signal path is clean — no dropped
sample buffers, no unintended DSP, no scaling errors — so that any on-air results
reflect RADE performance, not integration issues.

**Scope:** This procedure tests integration correctness only. It does not
evaluate speech quality, compare V1 vs V2, or directly assess radio hardware
performance. Any additional tests beyond this procedure should be agreed with
the RADE team before being submitted as results.

## Requirements

- The application must be able to export feature vectors at the RADE encoder
  input (Tx) and RADE decoder output (Rx) to disk files, for use with `loss.py`.
  See `rade_tx_wav -f` and `rade_rx_wav -f` in the rade_c repo, and the
  `rade_c_v2_wav` ctest, for worked examples.
- The signal path must have a transfer function of 1 in both directions —
  **no additional signal processing** (AGC, noise gate, resampler, EQ,
  compression) between WAV file input and RADE encoder input, or between
  RADE decoder output and the exported feature files.
- Tests must use `wav/all.wav` (56 seconds, 16 kHz mono) from this repository.
- `loss.py` must be from this repository (not a local copy) to ensure consistency.

## Test Levels

### Level 1 — Software Loopback (mandatory)

File in → application Tx → application Rx → file out, no hardware.

Establishes that the application's RADE signal path is correct in software
before any hardware is introduced.

**At least one of Level 2 or Level 3 is mandatory for hardware integrations.**

**Over The Air (OTA) tests are not part of this procedure.** The radio channel
introduces uncontrolled variables that make loss measurements unrepeatable.

### Level 2 — Over The Audio Cable / OTAC

Speech WAV → application → **sound card out → audio cable → sound card in** →
application → decoded WAV.

Tests the complete audio path including sound drivers, which are a common source
of bugs (dropped buffers, sample rate mismatches, bit depth errors).

### Level 3 — Over The Coax / OTC

Full RF path: application → DAC → HF Tx → attenuator → coax → attenuator →
HF Rx → ADC → application.

Tests the complete hardware chain.

## Step-by-Step Procedure

### Step 1 — Establish the current baseline

Re-run the following with the latest version of this repository to obtain the
current software-only baseline. Do not use a cached value — the baseline shifts
slightly between model versions.

```
cd ~/radae
lpcnet_demo -features wav/all.wav features_in.f32
python3 tx2.py 250725/checkpoints/checkpoint_epoch_200.pth features_in.f32 tx.f32
python3 rx2.py 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync tx.f32 features_rx.f32 --quiet
python3 loss.py features_in.f32 features_rx.f32 --clip_start 100 --clip_end 300
```

Example output (Python reference, `wav/all.wav`, model `250725`, commit `b549586`):
```
loss: 0.081 start: 224 acq_time:  1.24 s
```

Record the current baseline loss value and the git commit hash used
(e.g. `git log --oneline -1` → `cafebabe`). **A pass is within ±10% of the baseline.**

### Step 2 — Level 1: Software loopback

Run `wav/all.wav` through your application's full encode/decode pipeline with no
hardware, exporting feature vectors at encoder input and decoder output. Then:

```
python3 loss.py features_tx.f32 features_rx.f32 --clip_start 100 --clip_end 300
```

Pass criterion: loss within ±10% of baseline.

### Step 3 — Level 2: OTAC (hardware integrations)

Connect sound card output to sound card input via audio cable (no radio, no RF).
Run the same test as Step 2 with the audio routed through the hardware path.

Pass criterion: loss within ±10% of baseline.

### Step 4 — Level 3: OTC (SDR or full RF path)

Connect Tx and Rx via coax with appropriate attenuators. Run the same test.

Pass criterion: loss within ±10% of baseline.

## Submitting Results

Copy `doc/verification/template.md` to `doc/verification/<serial>-<application>.md`
(e.g. `001-freedv-gui.md`), fill it in, and submit as a PR or attach to the
relevant GitHub issue.

**Multi-platform applications** (Windows/Linux/Mac) must submit a separate
completed form for each platform, as sound hardware drivers differ.
