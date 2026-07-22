# RADE Integration Verification Report

## Application Under Test

| Field | Value |
|---|---|
| Application name | |
| Application version / git hash | |
| Platform (OS + version) | |
| Tester name / callsign | |
| Date | |
| radae repo commit hash | |
| rade_c repo commit hash | |

## Signal Path Declaration

- [ ] No additional signal processing (AGC, noise gate, resampler, EQ,
      compression) between WAV file input and RADE encoder input during
      this verification test

## Baseline Loss (Step 1)

Re-run with the current repository version before filling in this section.

| Field | Value |
|---|---|
| Baseline loss | |
| 10% tolerance window | baseline ± |

Command used:
```
(paste command here)
```

## Level 1 — Software Loopback (mandatory)

- [ ] Pass (loss within ±10% of baseline)

| Field | Value |
|---|---|
| Loss result | |

Command used / reproduction notes:
```
(paste command here)
```

## Level 2 — OTAC: Over The Audio Cable (mandatory for hardware integrations)

- [ ] Pass (loss within ±10% of baseline)
- [ ] N/A (software-only integration)

| Field | Value |
|---|---|
| Loss result | |
| Sound card (Tx) | |
| Sound card (Rx) | |
| Cable description | |

Photo of test setup:
_(attach or link)_

Reproduction notes:
```
(paste notes here)
```

## Level 3 — OTC: Over The Coax (optional)

- [ ] Pass (loss within ±10% of baseline)
- [ ] Not performed

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
| Level 1 — Software loopback | PASS / FAIL / N/A |
| Level 2 — OTAC | PASS / FAIL / N/A |
| Level 3 — OTC | PASS / FAIL / N/A |

Additional notes:
