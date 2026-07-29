# PI-KDA

Reference implementation of **PI-KDA: Physics-Informed Kernel Discriminant
Analysis for Stealthy and Overt Attack Detection in Autonomous Vehicles**.

PI-KDA is a two-layer detector that requires no learned prediction model:

- **Stage 1** evaluates 18 physics-informed residual channels (kinematic and
  dynamic identities any vehicle in motion must satisfy) on raw on-board sensor
  readings. `compute_consistency()` returns them in paper order: code index
  = Table I channel number - 1.
- **Layer 1 (overt)** runs a two-sided CUSUM independently on channels 7-10,
  thresholded at 1.1x the 0.995 benign quantile.
- **Layer 2 (stealthy)** feeds multi-scale rolling features into five
  pathway-specialized kernel Fisher discriminants (linear + RBF + Laplacian
  multi-kernel), followed by a drift-corrected CUSUM, scene-adaptive
  threshold scaling, and the ELITE confirmation gate.
- The two layers run in parallel; a trace is flagged as soon as either fires.

## Installation

```bash
pip install numpy pandas scipy numba joblib optuna
```

Tested with Python 3.10+.



## Running

Set `DATASET_DIRS` in `pi_kda.py`, then:

```bash
python pi_kda.py
```

On the first run the script trains: it calibrates Layer-1 thresholds on
benign training data alone, then for each pathway runs the hyper-parameter
search (Optuna TPE over kernel weights, exhaustive grid over CUSUM
allowance quantile and bootstrap block size), scoring every candidate by
the **full deployment decision path** (drift-corrected CUSUM +
scene-adaptive scaling + ELITE gate) so that validation F1 predicts
deployed F1. Trained models are cached in `pi_kda_trained_models.pkl`;
subsequent runs skip straight to evaluation.

Evaluation uses the paper's metric convention (Eq. 10, TOLERANCE = 0): an
alarm before the attack onset counts as a pre-attack false positive in
both the precision and the recall denominators.

## Paper-to-code mapping

**Pathways (Table II).**

| Paper | Code group | Trains on (code folder names) |
|-------|------------|-------------------------------|
| P_act | `ctrl`     | `stealthyACT1/ACT3/ACT4`      |
| P_odo | `wheel_v`  | `stealthyS10`                 |
| P_gi  | `gps_imu`  | `stealthyS4`, `stealthyS5`    |
| P_pv  | `gps_xy_v` | `stealthyS2`                  |
| P_pa  | `gps_xy_a` | `stealthyS3`                  |

Folder names predate the paper's final attack numbering, so codes differ
between the two; the table above is authoritative. `stealthyS1` (GNSS
lat/lon-only spoofing) is deliberately excluded from every training group
and serves as the unseen test-only type of Table III. Folders named
`attack-*` without `stealthy` are the overt types and are always test-only.

**Key constants.**

| Paper                                   | Code                          |
|-----------------------------------------|-------------------------------|
| Warm-up interval [50, 350]              | `WARMUP`                      |
| Layer-1 n1 = 10 consecutive frames      | `CONSEC_L1`                   |
| Layer-2 n2 = 5 consecutive frames       | `CONSEC`                      |
| ELITE gate: 50-frame sustain, 1.10 peak | `ELITE_SUSTAIN_MIN/_PEAK_MIN` |
| Active pathway: validation F1 >= 0.4    | `ACTIVE_F1_THRESHOLD`         |
| Rolling windows W = 20 / 80 / 160       | `CHANNEL_WINDOW`              |

  note    = {under review}
}
```
