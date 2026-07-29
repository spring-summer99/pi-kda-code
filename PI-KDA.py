"""
PI-KDA: Physics-Informed Kernel Discriminant Analysis for Stealthy and
Overt Attack Detection in Autonomous Vehicles.

Single-file reference implementation accompanying the paper.

Pipeline (paper Fig. 2 / Algorithm 1):
  Stage 1  compute_consistency()      -> 18 physics-informed residual channels
  Stage 2  Layer 1: per-channel two-sided CUSUM on 4 raw residuals (overt)
  Stage 3  Layer 2: pathway-specialized KFDA + drift-corrected CUSUM
           + scene-adaptive threshold + ELITE gate            (stealthy)
  Fusion   OR rule: a trace is flagged as soon as either layer fires.

compute_consistency() returns the 18 channels of Table I in paper order:
code index = paper channel number - 1. Layer 1 monitors indices 6-9
(paper channels 7-10). The five pathways of Table II map to the code
groups: P_act='ctrl', P_odo='wheel_v', P_gi='gps_imu', P_pv='gps_xy_v',
P_pa='gps_xy_a'.

All inputs are the *_used sensor values, i.e. what the vehicle actually
consumed (possibly attacker-controlled) -- never simulator ground truth.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import savgol_filter
import optuna

# ==========================================================================
# Configuration
# ==========================================================================
L = 4.8            # wheelbase (m)
K_US = 0.002       # understeer gradient
CONSEC = 5         # Layer-2 consecutive-exceedance requirement (n2)
CONSEC_L1 = 10     # Layer-1 consecutive-exceedance requirement (n1)
IGNORE_STARTUP = 50            # frames zeroed at trace start (2.5 s @ 20 Hz)
WARMUP = (50, 350)             # clean warm-up interval for normalization
ELITE_SUSTAIN_MIN = 50         # ELITE gate: frames above threshold in window
ELITE_PEAK_MIN = 1.10          # ELITE gate: peak / threshold ratio
ELITE_WINDOW = 600             # ELITE gate: evaluation window after alarm
ACTIVE_F1_THRESHOLD = 0.4      # pathway is active if validation F1 >= this

# Residual-channel subsets per pathway (indices into Table I, 0-based).
FEATURE_SETS = {
    'ctrl':     [7, 4, 2, 9, 0, 13, 12],           # P_act
    'wheel_v':  [10, 11, 5, 7],                    # P_odo
    'gps_imu':  [7, 1, 4, 0, 6, 15, 16],           # P_gi
    'gps_xy_v': [7, 2, 3, 14, 16, 15, 0, 6],       # P_pv
    'gps_xy_a': [7, 2, 3, 14, 16, 15, 0, 6],       # P_pa
}

# Which attack folder names train which pathway. stealthyS1 is absent from
# every group by design: it is the unseen, test-only type of Table III.
ATTACK_GROUPS = {
    'ctrl':     {'stealthyACT1', 'stealthyACT2', 'stealthyACT3'},
    'wheel_v':  {'stealthyS5'},
    'gps_imu':  {'stealthyS4'},
    'gps_xy_v': {'stealthyS2'},
    'gps_xy_a': {'stealthyS3'},
}
KFDA_GROUPS = list(FEATURE_SETS.keys())
ALL_TRAINABLE_ATTACKS = set().union(*ATTACK_GROUPS.values())

# Stealthy attack whitelist for the released evaluation scope; folders
# named attack-* without 'stealthy' are the overt types and are always
# test-only.
INCLUDE_ATTACKS = {
    'stealthyACT1', 'stealthyACT3', 'stealthyACT4',
    'stealthyS1', 'stealthyS2', 'stealthyS3',
    'stealthyS5', 'stealthyS10',
}

# Rolling aggregation window per channel (frames @ 20 Hz), Table I column
# "Win.": 20 = fast transients, 80 = medium, 160 = slow drifts.
CHANNEL_WINDOW = {
    0: 20, 1: 20, 2: 80, 3: 80, 4: 80, 5: 20, 6: 20, 7: 160, 8: 20,
    9: 80, 10: 160, 11: 80, 12: 160, 13: 80, 14: 80, 15: 160, 16: 80,
    17: 80,
}

DATASET_DIRS = [
    # Each root must contain benign/ and attack/ subdirectories, each
    # holding one folder per trace with curvature-consistency.csv and
    # savior_log.csv.
    "./dataset",
]
CACHE_FILE = "pi_kda_trained_models.pkl"


# ==========================================================================
# Data loading (sensor values only)
# ==========================================================================
def load_trace(csv_path):
    """Load one trace as the detector sees it in deployment.

    Position, heading, and speed come from the x_used/y_used/psi_used/
    v_used columns of savior_log.csv; accelerations, yaw rate, steering,
    and dt come from curvature-consistency.csv. Every column is a sensor
    value the vehicle actually consumed."""
    folder = os.path.dirname(csv_path)
    df_c = pd.read_csv(csv_path)
    df_s = pd.read_csv(os.path.join(folder, 'savior_log.csv'))
    n = min(len(df_c), len(df_s))
    return pd.DataFrame({
        'x':      df_s['x_used'].to_numpy(dtype=float)[:n],
        'y':      df_s['y_used'].to_numpy(dtype=float)[:n],
        'psi':    df_s['psi_used'].to_numpy(dtype=float)[:n],
        'v':      df_s['v_used'].to_numpy(dtype=float)[:n],
        'acc_x':  df_c['acc_x_used'].to_numpy(dtype=float)[:n],
        'acc_y':  df_c['acc_y_used'].to_numpy(dtype=float)[:n],
        'gyro_z': df_c['gyro_z_used'].to_numpy(dtype=float)[:n],
        'delta':  df_c['delta_rad'].to_numpy(dtype=float)[:n],
        'dt':     float(df_c['dt'].iloc[0]),
    })


# ==========================================================================
# Stage 1: the 18 physics-informed residual channels (Table I)
# ==========================================================================
@njit
def compute_robust_path_yaw_rate(x, y, dt):
    """GNSS-path yaw rate r_geo via central differences + phase unwrap."""
    n = len(x)
    r_geo = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(1, n - 1):
        vx[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
        vy[i] = (y[i + 1] - y[i - 1]) / (2 * dt)
    vx[0], vy[0] = vx[1], vy[1]
    vx[-1], vy[-1] = vx[-2], vy[-2]

    psi = np.arctan2(vy, vx)
    for i in range(1, n):
        d = psi[i] - psi[i - 1]
        if d > np.pi:
            psi[i] -= 2 * np.pi
        elif d < -np.pi:
            psi[i] += 2 * np.pi
    for i in range(1, n - 1):
        r_geo[i] = (psi[i + 1] - psi[i - 1]) / (2 * dt)
    r_geo[0], r_geo[-1] = r_geo[1], r_geo[-2]
    return r_geo


def compute_consistency(df, steer_scale=1.0):
    """Evaluate the kinematic/dynamic identities of Section II-A on raw
    sensor readings; return the 18 residual channels of Table I in paper
    order (index = paper channel - 1)."""
    x = df["x"].to_numpy(dtype=float, copy=True)
    y = df["y"].to_numpy(dtype=float, copy=True)
    psi = df["psi"].to_numpy(dtype=float, copy=True)
    v = df["v"].to_numpy(dtype=float, copy=True)
    ax = df["acc_x"].to_numpy(dtype=float, copy=True)
    ay = df["acc_y"].to_numpy(dtype=float, copy=True)
    r = df["gyro_z"].to_numpy(dtype=float, copy=True)
    delta = df["delta"].to_numpy(dtype=float, copy=True)
    dt = float(df["dt"].iloc[0])

    # -- sanitation: NaN/Inf, simulator start-up spikes, constant series --
    for arr in (x, y, psi, v, ax, ay, r, delta):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(min(10, len(ax))):
        if np.abs(ax[i]) > 50.0:
            ax[i] = 0.0
        if np.abs(ay[i]) > 50.0:
            ay[i] = 0.0
    rng = np.random.default_rng(0)
    if np.ptp(r) < 1e-12:
        r = r + rng.normal(0, 1e-9, r.shape)
    if np.ptp(ay) < 1e-12:
        ay = ay + rng.normal(0, 1e-9, ay.shape)
    if np.ptp(v) < 1e-12:
        v = v + rng.normal(0, 1e-9, v.shape)

    is_moving = np.abs(v) > 0.1
    v_safe = np.where(is_moving, v, 1.0)

    # -- GNSS-derived kinematics --
    vx_g = np.gradient(x, dt)
    vy_g = np.gradient(y, dt)
    vn = vx_g * np.cos(psi) + vy_g * np.sin(psi)
    ve = -vx_g * np.sin(psi) + vy_g * np.cos(psi)
    vn = np.where(is_moving, vn, 1e-6)
    ve = np.where(is_moving, ve, 0.0)
    beta = np.unwrap(np.arctan2(ve, vn))     # body-frame velocity direction

    if len(v) >= 9:
        dv = savgol_filter(v, 9, 2, deriv=1, delta=dt)
    else:
        dv = np.gradient(v, dt)

    r_geo = compute_robust_path_yaw_rate(x, y, dt)

    # ch 1: longitudinal force  a_x - dv/dt
    eps_long = ax - dv
    # ch 2: lateral balance (signed)  a_y - v*r
    eps_lat = ay - (v * r)
    # ch 3: steering geometry  v*tan(d) - L*r_geo - Kus*a_y*v
    delta_eff = delta * steer_scale
    eps_steer_geo = v_safe * np.tan(delta_eff) - L * r_geo - K_US * ay * v_safe
    # ch 4: path-vs-gyro yaw  r_geo - r
    eps_path_gyro = r_geo - r
    # ch 5: steering dynamics  v*tan(d) - L*r - Kus*a_y*v
    eps_steer_dyn = v_safe * np.tan(delta_eff) - L * r - K_US * ay * v_safe
    for e in (eps_long, eps_lat, eps_steer_geo, eps_path_gyro, eps_steer_dyn):
        e[~is_moving] = 0.0

    # ch 6: velocity-integral step  |a_x*dt - dv|  (ungated)
    delta_v = np.diff(v, prepend=v[0])
    eps_vel_step = np.abs(ax * dt - delta_v)
    # ch 7: velocity-integral step, gated  (Layer 1)
    eps_vel_step_g = np.abs(ax * dt - delta_v)
    eps_vel_step_g[~is_moving] = 0.0

    # ch 8: dead-reckoning position drift  ||p - p_hat||  (Layer 1)
    n = len(x)
    x_est = np.copy(x)
    y_est = np.copy(y)
    for i in range(IGNORE_STARTUP + 1, n):
        x_est[i] = x_est[i - 1] + v[i] * np.cos(psi[i]) * dt
        y_est[i] = y_est[i - 1] + v[i] * np.sin(psi[i]) * dt
    eps_pos_drift = np.sqrt((x - x_est) ** 2 + (y - y_est) ** 2)
    eps_pos_drift[~is_moving] = 0.0

    # ch 9: lateral-balance magnitude  |a_y - v*r|  (Layer 1)
    lat_res = ay - v * r
    lat_res[~is_moving] = 0.0
    eps_lat_mag = np.abs(lat_res)

    # ch 10: a_x - dv correlation anomaly  (Layer 1)
    WIN_CORR = 50
    dv_dt = np.diff(v, prepend=v[0]) / dt
    _ax_s = pd.Series(ax)
    corr_ax_dv = _ax_s.rolling(WIN_CORR, min_periods=2).corr(
        pd.Series(dv_dt)).fillna(1.0).to_numpy()
    eps_ax_dv = 1.0 - corr_ax_dv

    # ch 11: speed autocorrelation (lag 50)
    _v_s = pd.Series(v)
    eps_v_autoc50 = np.abs(_v_s.rolling(100, min_periods=20).corr(
        _v_s.shift(50)).fillna(0.0).to_numpy())

    # ch 12: gyro autocorrelation (lag 1). A frozen/zeroed gyro yields NaN
    # rolling corr (zero variance); filling with 0 marks it anomalous.
    _r_s = pd.Series(r)
    eps_gyro_autoc1 = 1.0 - np.abs(_r_s.rolling(20, min_periods=5).corr(
        _r_s.shift(1)).fillna(0.0).to_numpy())

    # ch 13: gyro freeze ratio  max(0, 1 - sigma_20 / sigma_200)
    _r_std_s = _r_s.rolling(20, min_periods=5).std().fillna(0.0).to_numpy()
    _r_std_l = _r_s.rolling(200, min_periods=20).std().fillna(1e-4).to_numpy()
    _r_std_l = np.where(_r_std_l < 1e-4, 1e-4, _r_std_l)
    eps_gyro_freeze = np.maximum(0.0, 1.0 - _r_std_s / _r_std_l)

    # ch 14: longitudinal x steering joint correlation  |rho_60(eps1, eps5)|
    eps_long_steer = np.abs(pd.Series(eps_long).rolling(60, min_periods=10)
                            .corr(pd.Series(eps_steer_dyn)).fillna(0.0).to_numpy())

    # ch 15: heading vs path tangent  |psi - beta| (unwrapped)
    eps_psi_path = np.abs(np.unwrap(psi) - np.unwrap(beta))

    # ch 16: GNSS vs IMU-integrated speed
    v_gps = np.sqrt(vx_g ** 2 + vy_g ** 2)
    v_int_imu = v[0] + np.cumsum(ax) * dt
    eps_gps_imu_v = np.abs(v_gps - v_int_imu)
    eps_gps_imu_v[~is_moving] = 0.0

    # ch 17: turn-weighted yaw  |r_geo - r| * w_turn
    rgeo_smooth = pd.Series(np.abs(r_geo)).rolling(
        80, min_periods=10).mean().fillna(0).to_numpy()
    turn_w = np.minimum(1.0, rgeo_smooth / 0.05)
    eps_yaw_turn = np.abs(r_geo - r) * turn_w
    eps_yaw_turn[~is_moving] = 0.0

    # ch 18: a_x autocorrelation (lag 40)
    eps_ax_autoc40 = np.abs(_ax_s.rolling(80, min_periods=20).corr(
        _ax_s.shift(40)).fillna(0.0).to_numpy())

    channels = [eps_long, eps_lat, eps_steer_geo, eps_path_gyro,
                eps_steer_dyn, eps_vel_step, eps_vel_step_g, eps_pos_drift,
                eps_lat_mag, eps_ax_dv, eps_v_autoc50, eps_gyro_autoc1,
                eps_gyro_freeze, eps_long_steer, eps_psi_path,
                eps_gps_imu_v, eps_yaw_turn, eps_ax_autoc40]

    if len(v) > IGNORE_STARTUP:
        for e in channels:
            e[:IGNORE_STARTUP] = 0.0
    return channels


# ==========================================================================
# Feature construction: multi-scale rolling stats + warm-up normalization
# ==========================================================================
def robust_normalize_iqr(G_raw, warmup_start=50, warmup_end=350, clip=15.0):
    """Scene-adaptive local normalization (paper Eq. 8): center each channel
    at its warm-up median and scale by its warm-up IQR; a channel whose
    warm-up spread is numerically zero is left unscaled (unit denominator).
    """
    ws_, we_ = warmup_start, min(warmup_end, len(G_raw))
    if we_ <= ws_:
        ws_ = 0
    G_w = G_raw[ws_:we_]
    lmed = np.median(G_w, axis=0)
    liqr = np.percentile(G_w, 75, axis=0) - np.percentile(G_w, 25, axis=0)
    liqr = np.where(liqr < 1e-6, 1.0, liqr)
    return np.clip((G_raw - lmed) / liqr, -clip, clip)


def stack_residuals(eps_list, feature_group):
    """Build the 2D-dim feature [rolling mean; rolling std] for one pathway.
    Each channel uses its physically matched window (CHANNEL_WINDOW)."""
    chosen_idx = FEATURE_SETS[feature_group]
    G_raw = np.vstack([eps_list[i] for i in chosen_idx]).T
    pd_G = pd.DataFrame(G_raw)

    means = np.zeros_like(G_raw)
    stds = np.zeros_like(G_raw)
    for col, ch in enumerate(chosen_idx):
        ws = CHANNEL_WINDOW[ch]
        col_s = pd_G.iloc[:, col]
        means[:, col] = col_s.rolling(window=ws, min_periods=1).mean().to_numpy()
        stds[:, col] = col_s.rolling(window=ws, min_periods=1).std().fillna(0).to_numpy()
    return np.hstack([means, stds])


def calibrate_steering_ratio(file_list):
    """Estimate the steering-wheel-to-road-wheel ratio from benign traces."""
    ratios = []
    for fp in file_list:
        df = load_trace(fp)
        v = df["v"].to_numpy()
        r = df["gyro_z"].to_numpy()
        delta = df["delta"].to_numpy()
        mask = (np.abs(v) > 2.0) & (np.abs(delta) > 0.05)
        if np.sum(mask) > 10:
            ratios.extend((r[mask] * L) / (v[mask] * np.tan(delta[mask])))
    return float(np.median(ratios)) if ratios else 1.0


def build_benign_matrices(file_list, steer_scale, feature_group):
    eps_all, G_list = [], []
    for fp in file_list:
        eps = compute_consistency(load_trace(fp), steer_scale)
        eps_all.append(eps)
        G = robust_normalize_iqr(stack_residuals(eps, feature_group))
        G = G[IGNORE_STARTUP:] if len(G) > IGNORE_STARTUP else G
        G = G[::4]
        G_list.append(G[np.isfinite(G).all(axis=1)])
    return eps_all, np.vstack(G_list)


def build_attack_train_matrix(file_list, steer_scale, attack_start, feature_group):
    """Attack training samples: the first 600 frames after onset (signal
    growth phase), subsampled 1-in-4."""
    G_list = []
    for fp in file_list:
        eps = compute_consistency(load_trace(fp), steer_scale)
        G = robust_normalize_iqr(stack_residuals(eps, feature_group))
        atk = attack_start.get(fp, 402)
        end = min(atk + ELITE_WINDOW, len(G))
        G = G[atk:end] if len(G) > atk else G[len(G) // 2:]
        G = G[::4]
        G_list.append(G[np.isfinite(G).all(axis=1)])
    return np.vstack(G_list) if G_list else np.zeros((0, 0))


# ==========================================================================
# CUSUM machinery
# ==========================================================================
@njit
def cusum_1d(residuals, k, b):
    """Two-sided CUSUM on one raw residual channel (paper Eq. 7).
    k: benign median; b: allowance. Returns max(C+, C-) per frame."""
    n = len(residuals)
    cp = np.zeros(n)
    cn = np.zeros(n)
    for i in range(1, n):
        if not np.isfinite(residuals[i]):
            cp[i], cn[i] = cp[i - 1], cn[i - 1]
            continue
        z = residuals[i] - k
        cp[i] = max(0.0, cp[i - 1] + z - b)
        cn[i] = max(0.0, cn[i - 1] - z - b)
    return np.maximum(cp, cn)


@njit
def adaptive_cusum_engine(s, v, k_base, b_up, b_dn, h_up, h_dn, consec, v_ref):
    """Velocity-scaled two-sided CUSUM on a KFDA score series; returns the
    first alarm index (-1 if none) plus the full C+/C- trajectories."""
    cp, cn = 0.0, 0.0
    run_up, run_dn = 0, 0
    T = len(s)
    cp_hist = np.zeros(T)
    cn_hist = np.zeros(T)
    alarm_idx = -1
    for i in range(T):
        if not np.isfinite(s[i]):
            if i > 0:
                cp_hist[i], cn_hist[i] = cp_hist[i - 1], cn_hist[i - 1]
            continue
        k_adaptive = k_base * (1.0 + 0.05 * abs(v[i]) / v_ref)
        z = s[i] - k_adaptive
        cp = max(0.0, cp + z - b_up)
        cn = max(0.0, cn - z - b_dn)
        cp_hist[i] = cp
        cn_hist[i] = cn
        run_up = run_up + 1 if cp > h_up else 0
        run_dn = run_dn + 1 if cn > h_dn else 0
        if (run_up >= consec or run_dn >= consec) and alarm_idx == -1:
            alarm_idx = i - consec + 1
    return alarm_idx, cp_hist, cn_hist


@njit
def adaptive_cusum_two_sided(s, v, k_base, b_up, b_dn, v_ref):
    T = len(s)
    cp = np.zeros(T)
    cn = np.zeros(T)
    for i in range(1, T):
        if not np.isfinite(s[i]):
            cp[i], cn[i] = cp[i - 1], cn[i - 1]
            continue
        k_adaptive = k_base * (1.0 + 0.05 * abs(v[i]) / v_ref)
        z = s[i] - k_adaptive
        cp[i] = max(0.0, cp[i - 1] + z - b_up)
        cn[i] = max(0.0, cn[i - 1] - z - b_dn)
    return cp, cn


@njit
def adaptive_cusum_max_two_sided(s, v, k_base, b_up, b_dn, v_ref):
    cp_curr, cn_curr = 0.0, 0.0
    max_cp, max_cn = 0.0, 0.0
    for i in range(len(s)):
        if not np.isfinite(s[i]):
            continue
        k_adaptive = k_base * (1.0 + 0.05 * abs(v[i]) / v_ref)
        z = s[i] - k_adaptive
        cp_curr = max(0.0, cp_curr + z - b_up)
        cn_curr = max(0.0, cn_curr - z - b_dn)
        if cp_curr > max_cp:
            max_cp = cp_curr
        if cn_curr > max_cn:
            max_cn = cn_curr
    return max_cp, max_cn


def _find_consec_violation(viol_indices, consec):
    """First index of `consec` consecutive violations (1e9 if none)."""
    if len(viol_indices) < consec:
        return int(1e9)
    for i in range(len(viol_indices) - consec + 1):
        if viol_indices[i + consec - 1] - viol_indices[i] == consec - 1:
            return int(viol_indices[i])
    return int(1e9)


def cusum_alarm_drift_corrected(s, v, k_proj, b_up, b_dn, h_up, h_dn, consec,
                                v_ref, return_gate_stats=False):
    """Drift-corrected sequential test used identically in training model
    selection and deployment (train-deploy alignment).

    Phase I estimates the linear CUSUM growth rate over the clean warm-up
    segment; Phase II subtracts that trend from the whole trajectory
    (clipped at zero) before thresholding. With return_gate_stats=True, the
    ELITE-gate statistics (sustain frames, peak ratio) are also returned."""
    INF_H = 1e12
    _, cp_h, cn_h = adaptive_cusum_engine(s, v, k_proj, b_up, b_dn,
                                          INF_H, INF_H, consec, v_ref)
    warm_s, warm_e = WARMUP[0], min(WARMUP[1], len(cp_h))
    if warm_e > warm_s + 20:
        w_len = warm_e - warm_s
        drift_up = max(0.0, (cp_h[warm_e - 1] - cp_h[warm_s]) / w_len)
        drift_dn = max(0.0, (cn_h[warm_e - 1] - cn_h[warm_s]) / w_len)
    else:
        drift_up = drift_dn = 0.0
    t_off = np.maximum(0.0, np.arange(len(cp_h), dtype=float) - warm_s)
    cp_corr = np.maximum(0.0, cp_h - drift_up * t_off)
    cn_corr = np.maximum(0.0, cn_h - drift_dn * t_off)

    viol_up = np.where(cp_corr > h_up)[0]
    viol_dn = np.where(cn_corr > h_dn)[0]
    viol_up = viol_up[viol_up > IGNORE_STARTUP]
    viol_dn = viol_dn[viol_dn > IGNORE_STARTUP]
    alarm = int(min(_find_consec_violation(viol_up, consec),
                    _find_consec_violation(viol_dn, consec)))
    if alarm >= int(1e9):
        alarm = -1
    if not return_gate_stats:
        return alarm

    if alarm > 0:
        end = min(alarm + ELITE_WINDOW, len(cp_corr))
        sustain = max(int((cp_corr[alarm:end] > h_up).sum()),
                      int((cn_corr[alarm:end] > h_dn).sum()))
        peak_ratio = max(float(cp_corr[alarm:end].max()) / max(h_up, 1e-6),
                         float(cn_corr[alarm:end].max()) / max(h_dn, 1e-6))
    else:
        sustain, peak_ratio = 0, 0.0
    return alarm, sustain, peak_ratio


def calibrate_layer1_thresholds(eps_b_list, info_base, QD=0.1,
                                Q_1B=0.995, SF_1B=1.1):
    """Layer-1 calibration on benign data alone. For each of the four
    monitored channels (indices 6-9 = paper channels 7-10): k = benign
    median, b = low quantile of positive deviations, alarm threshold =
    0.995 quantile of pooled benign CUSUM scores x 1.1."""
    ISU = IGNORE_STARTUP
    info = info_base.copy()
    for ch_idx, name in [(6, 'vel'), (7, 'pos'), (8, 'lat'), (9, 'axdv')]:
        br = np.concatenate([e[ch_idx][ISU:] for e in eps_b_list])
        br = br[np.isfinite(br)]
        k = float(np.median(br))
        pos = br[br > k] - k
        b = float(np.quantile(pos, QD)) if len(pos) > 0 else 0.01
        all_c = np.concatenate([cusum_1d(e[ch_idx][ISU:], k, b)
                                for e in eps_b_list])
        info[f'cusum_k_{name}'] = k
        info[f'cusum_b_{name}'] = b
        info[f'cusum_h_{name}'] = float(np.quantile(all_c, Q_1B)) * SF_1B
    return info


# ==========================================================================
# Stage 3: kernel Fisher discriminant analysis
# ==========================================================================
def _pairwise_kernel(X, Y=None, weight_rbf=1.0, weight_lap=0.0):
    """Multi-kernel Gram matrix (paper Section IV-C):
        K = w_lin*K_lin + w_rbf*K_rbf + w_lap*K_lap,  w_lin = 1 - w_rbf - w_lap.
    RBF/Laplacian bandwidths use the median heuristic (no manual tuning)."""
    X = np.ascontiguousarray(np.asarray(X, float))
    Y = X if Y is None else np.ascontiguousarray(np.asarray(Y, float))

    K_lin = X @ Y.T
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True)
    D2 = np.maximum(XX + YY.T - 2.0 * (X @ Y.T), 0.0)
    tri = D2[np.triu_indices_from(D2, k=1)] if X is Y else D2
    med2 = float(np.median(tri[~np.isnan(tri)])) if tri.size else 1.0
    gamma = 1.0 / (2.0 * (med2 + 1e-12))
    K_rbf = np.exp(-gamma * D2)

    w_r = max(0.0, float(weight_rbf))
    w_l = max(0.0, float(weight_lap))
    if w_r + w_l > 1.0:
        w_r, w_l = w_r / (w_r + w_l), w_l / (w_r + w_l)
    w_lin = max(0.0, 1.0 - w_r - w_l)

    K = w_lin * K_lin + w_r * K_rbf

    if w_l > 1e-9:
        n, m = X.shape[0], Y.shape[0]
        K_lap = np.empty((n, m), dtype=float)
        block = 128        # blockwise L1 distances to bound peak memory
        n_samp = min(256, n)
        idx_s = (np.random.default_rng(0).choice(n, size=n_samp, replace=False)
                 if n > n_samp else np.arange(n))
        D1_samp = np.abs(X[idx_s, None, :] - X[None, idx_s, :]).sum(axis=2)
        med1 = float(np.median(D1_samp[D1_samp > 0])) if (D1_samp > 0).any() else 1.0
        gamma_L = 1.0 / (med1 + 1e-12)
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            D1 = np.abs(X[i0:i1, None, :] - Y[None, :, :]).sum(axis=2)
            K_lap[i0:i1] = np.exp(-gamma_L * D1)
        K = K + w_l * K_lap
    return K


def kfda_projection(G_b, G_a, weight_rbf=0.5, weight_lap=0.0, reg=1e-6):
    """Two-class kernel Fisher discriminant (paper Eq. 9). Returns the dual
    coefficients alpha, the pooled training matrix, and kernel params."""
    Xb = np.asarray(G_b, float)
    Xa = np.asarray(G_a, float)
    n_b, n_a = Xb.shape[0], Xa.shape[0]
    X = np.vstack([Xb, Xa])
    K = _pairwise_kernel(X, weight_rbf=weight_rbf, weight_lap=weight_lap)
    idx_b = np.arange(n_b)
    idx_a = np.arange(n_b, n_b + n_a)
    K_b = K[np.ix_(idx_b, np.arange(n_b + n_a))]
    K_a = K[np.ix_(idx_a, np.arange(n_b + n_a))]
    Hb = np.eye(n_b) - np.ones((n_b, n_b)) / max(n_b, 1)
    Ha = np.eye(n_a) - np.ones((n_a, n_a)) / max(n_a, 1)
    N_mat = K_b.T @ Hb @ K_b + K_a.T @ Ha @ K_a + reg * np.eye(n_b + n_a)
    m = np.zeros((n_b + n_a, 1))
    m[idx_b, 0] = 1.0 / max(n_b, 1)
    m[idx_a, 0] = -1.0 / max(n_a, 1)
    alpha = np.linalg.solve(N_mat, K @ m).reshape(-1)
    alpha = alpha / (np.linalg.norm(K @ alpha) + 1e-12)
    return alpha, X, dict(weight_rbf=weight_rbf, weight_lap=weight_lap)


def kfda_score_time_series(G_time, trainX, alpha, params):
    s = np.full(len(G_time), np.nan)
    mask = np.isfinite(G_time).all(axis=1)
    if np.any(mask):
        K_t = _pairwise_kernel(trainX, G_time[mask], **params)
        s[mask] = alpha @ K_t
    return s


def score_trace(fp, trainX, alpha, params, steer_scale, feature_group):
    """Full per-trace scoring path: residuals -> features -> warm-up
    normalization -> KFDA projection -> warm-up z-scoring."""
    eps = compute_consistency(load_trace(fp), steer_scale)
    G = robust_normalize_iqr(stack_residuals(eps, feature_group))
    s = kfda_score_time_series(G, trainX, alpha, params)
    if len(s) > IGNORE_STARTUP:
        se = min(WARMUP[1], len(s))
        rm = np.median(s[IGNORE_STARTUP:se])
        rs = np.std(s[IGNORE_STARTUP:se]) + 1e-10
        s = (s - rm) / rs
        s[:IGNORE_STARTUP] = 0.0
    return s


# ==========================================================================
# Layer-2 CUSUM calibration (block bootstrap) and scene adaptation
# ==========================================================================
def calibrate_cusum_for_kfda(S_BENIGN, V_BENIGN, v_ref, q_deadzone=0.7,
                             block_size=100, n_bootstrap=500,
                             fpr_target=0.0005, S_ATTACK=None):
    """Calibrate one pathway's CUSUM: the center line k is the midpoint of
    the benign and attack score medians; allowances b+/b- are benign score
    quantiles; thresholds h+/h- come from a moving-block bootstrap
    (fixed-length half-overlapping blocks) of the benign scores, taking a
    high quantile of the simulated CUSUM maxima."""
    np.random.seed(42)
    all_s = np.concatenate([s[np.isfinite(s)] for s in S_BENIGN.values()])
    k_benign = float(np.median(all_s))

    if S_ATTACK:
        atk = [s[a:][np.isfinite(s[a:])] for s, a in S_ATTACK if a < len(s)]
        atk = [p for p in atk if len(p) > 0]
        k_proj = (0.5 * (k_benign + float(np.median(np.concatenate(atk))))
                  if atk else k_benign)
    else:
        k_proj = k_benign

    pos_s = all_s[all_s > k_proj] - k_proj
    neg_s = -(all_s[all_s < k_proj] - k_proj)
    b_u = float(np.quantile(pos_s, q_deadzone)) if len(pos_s) > 0 else 0.1
    b_d = float(np.quantile(neg_s, q_deadzone)) if len(neg_s) > 0 else 0.1

    all_v = np.concatenate([v for v in V_BENIGN.values()])
    n = len(all_s)
    blks_s = [all_s[i:i + block_size] for i in range(0, n - block_size, block_size // 2)]
    blks_v = [all_v[i:i + block_size] for i in range(0, n - block_size, block_size // 2)]
    if len(blks_s) < 2:
        return {'k_proj': k_proj, 'B_UP': b_u, 'B_DN': b_d,
                'h_up': 10.0, 'h_down': 10.0, 'v_ref': v_ref}

    target_length = int(np.median([len(s) for s in S_BENIGN.values()]))
    n_blocks_per_sim = max(1, target_length // block_size)
    cp_list, cn_list = [], []
    for _ in range(n_bootstrap):
        chosen = np.random.choice(len(blks_s), size=n_blocks_per_sim, replace=True)
        s_sim = np.concatenate([blks_s[i] for i in chosen])
        v_sim = np.concatenate([blks_v[i] for i in chosen])
        mp, mn = adaptive_cusum_max_two_sided(s_sim, v_sim, k_proj, b_u, b_d, v_ref)
        cp_list.append(mp)
        cn_list.append(mn)

    return {'k_proj': k_proj, 'B_UP': b_u, 'B_DN': b_d,
            'h_up': float(np.quantile(cp_list, 1.0 - fpr_target)),
            'h_down': float(np.quantile(cn_list, 1.0 - fpr_target)),
            'v_ref': v_ref}


def compute_scene_calibration_factor(s_test_warmup, V_BENIGN_CACHE,
                                     S_BENIGN_CACHE, v_test_warmup, info,
                                     warmup_start=WARMUP[0],
                                     warmup_end=WARMUP[1]):
    """Scene-adaptive threshold scaling: compare this trace's warm-up CUSUM
    noise (99th pct of the accumulator) to the median warm-up noise of the
    training benign traces. Composed with the [1.0, 4.0] clip at the call
    site, the effective range is [1.0, 1.8]: adaptation can only raise
    thresholds, never lower them."""
    cp_t, cn_t = adaptive_cusum_two_sided(
        s_test_warmup, v_test_warmup,
        info['k_proj'], info['B_UP'], info['B_DN'], info['v_ref'])
    test_noise = max(np.percentile(cp_t, 99) if len(cp_t) else 0.0,
                     np.percentile(cn_t, 99) if len(cn_t) else 0.0)

    benign_noise = []
    for fp, s_b in S_BENIGN_CACHE.items():
        v_b = V_BENIGN_CACHE[fp]
        end_idx = min(warmup_end, len(s_b))
        if end_idx - warmup_start < 50:
            continue
        cp_b, cn_b = adaptive_cusum_two_sided(
            s_b[warmup_start:end_idx], v_b[warmup_start:end_idx],
            info['k_proj'], info['B_UP'], info['B_DN'], info['v_ref'])
        benign_noise.append(max(np.percentile(cp_b, 99) if len(cp_b) else 0.0,
                                np.percentile(cn_b, 99) if len(cn_b) else 0.0))

    benign_med = np.median(benign_noise) if benign_noise else 1.0
    factor = test_noise / benign_med if benign_med > 1e-6 else 1.0
    return float(np.clip(factor, 0.6, 1.8))


# ==========================================================================
# Ensemble detector (paper Algorithm 1)
# ==========================================================================
def ensemble_detector(df, KFDA_MODELS, info, steer_scale,
                      S_BENIGN_CACHE=None, V_BENIGN_CACHE=None,
                      active_groups=None):
    """Run both layers on one trace.

    Returns (layer1_alarm_idx, layer2_alarm_idx, per_group_alarms), each
    -1 if the corresponding layer never fires. The trace-level alarm is
    the OR fusion min() of the two."""
    v = df["v"].to_numpy()
    eps_list = compute_consistency(df, steer_scale)

    # ---- Layer 1: per-channel CUSUM on 4 raw residuals (ch 7-10) ----
    l1_alarms = []
    for ch_idx, name in [(6, 'vel'), (7, 'pos'), (8, 'lat'), (9, 'axdv')]:
        eps_ch = eps_list[ch_idx]
        cusum_full = np.zeros(len(eps_ch))
        cusum_full[IGNORE_STARTUP:] = cusum_1d(
            eps_ch[IGNORE_STARTUP:],
            info[f'cusum_k_{name}'], info[f'cusum_b_{name}'])
        viol = np.where(cusum_full > info[f'cusum_h_{name}'])[0]
        viol = viol[viol > IGNORE_STARTUP]
        l1_alarms.append(_find_consec_violation(viol, CONSEC_L1))
    layer1_alarm = min(l1_alarms)
    layer1_alarm = -1 if layer1_alarm >= int(1e9) else layer1_alarm

    # ---- Layer 2: pathway-specialized KFDA ----
    v_ref = info['v_ref']
    per_group_alarms, per_group_sustain, per_group_peak = {}, {}, {}

    for group_name in KFDA_GROUPS:
        model = KFDA_MODELS.get(group_name)
        if model is None:
            per_group_alarms[group_name] = -1
            per_group_sustain[group_name] = 0
            per_group_peak[group_name] = 0.0
            continue

        G = robust_normalize_iqr(stack_residuals(eps_list, group_name))
        s = kfda_score_time_series(G, model['trainX'], model['alpha'],
                                   model['params'])
        if len(s) > IGNORE_STARTUP:
            se = min(WARMUP[1], len(s))
            rm = np.median(s[IGNORE_STARTUP:se])
            rs = np.std(s[IGNORE_STARTUP:se]) + 1e-10
            s = (s - rm) / rs
            s[:IGNORE_STARTUP] = 0.0

        kparams = info['kfda'][group_name]

        # Scene-adaptive threshold scaling.
        h_up_eff, h_dn_eff = kparams['h_up'], kparams['h_down']
        if S_BENIGN_CACHE is not None and group_name in S_BENIGN_CACHE \
                and V_BENIGN_CACHE is not None:
            warmup_s = s[IGNORE_STARTUP:min(WARMUP[1], len(s))]
            warmup_v = v[IGNORE_STARTUP:min(WARMUP[1], len(v))]
            if len(warmup_s) > 10:
                factor = compute_scene_calibration_factor(
                    warmup_s, V_BENIGN_CACHE, S_BENIGN_CACHE[group_name],
                    warmup_v, kparams, warmup_start=0, warmup_end=len(warmup_s))
                scale = float(np.clip(factor, 1.0, 4.0))
                h_up_eff = kparams['h_up'] * scale
                h_dn_eff = kparams['h_down'] * scale

        alarm_g, sustain, peak = cusum_alarm_drift_corrected(
            s, v, kparams['k_proj'], kparams['B_UP'], kparams['B_DN'],
            h_up_eff, h_dn_eff, CONSEC, v_ref, return_gate_stats=True)
        per_group_alarms[group_name] = alarm_g
        per_group_sustain[group_name] = sustain
        per_group_peak[group_name] = peak

    # ---- Layer-2 decision: ELITE gate over the active pathway set ----
    if active_groups is None:
        active_groups = set(per_group_alarms.keys())

    def _passes(g):
        return (per_group_alarms.get(g, -1) > 0
                and per_group_sustain.get(g, 0) >= ELITE_SUSTAIN_MIN
                and per_group_peak.get(g, 0.0) >= ELITE_PEAK_MIN)

    active_alarms = sorted(per_group_alarms[g] for g in active_groups if _passes(g))
    layer2_alarm = int(active_alarms[0]) if active_alarms else -1

    return layer1_alarm, layer2_alarm, per_group_alarms


# ==========================================================================
# Per-pathway hyper-parameter search (train-deploy alignment)
# --------------------------------------------------------------------------
# Every candidate configuration is scored by running the FULL deployment
# decision path (drift-corrected CUSUM + scene-adaptive scaling + ELITE
# gate) on the validation traces, so validation F1 predicts deployed F1.
# Outer loop: Optuna TPE over (weight_rbf, weight_lap).
# Inner loop: exhaustive grid over (q_deadzone, block_size).
# ==========================================================================
def search_best_hyperparams_for_group(group_name, BENIGN_TRAIN, BENIGN_VAL,
                                      ATK_TRAIN, ATK_VAL, steer_scale, v_ref,
                                      attack_start,
                                      weight_rbfs, q_deadzones, block_sizes,
                                      n_bootstrap, n_trials):
    print(f"\n[SEARCH] pathway '{group_name}': "
          f"{len(ATK_TRAIN)} train / {len(ATK_VAL)} val attack traces")

    # Split benign validation: first half calibrates thresholds, second
    # half evaluates FPR, avoiding fit-and-score on the same traces.
    n_calib = max(1, len(BENIGN_VAL) // 2)
    val_calib, val_eval = BENIGN_VAL[:n_calib], BENIGN_VAL[n_calib:] or BENIGN_VAL[:n_calib]
    V_calib = {fp: load_trace(fp)["v"].to_numpy() for fp in val_calib}
    V_eval = {fp: load_trace(fp)["v"].to_numpy() for fp in val_eval}
    V_atk = {fp: load_trace(fp)["v"].to_numpy() for fp in ATK_VAL}
    V_train_ref = {fp: load_trace(fp)["v"].to_numpy() for fp in BENIGN_TRAIN}

    # Training matrices are configuration-independent: build once.
    _, G_b_tr = build_benign_matrices(BENIGN_TRAIN, steer_scale, group_name)
    G_a_tr = build_attack_train_matrix(ATK_TRAIN, steer_scale, attack_start, group_name)
    np.random.seed(42)
    if len(G_b_tr) > 6000:
        G_b_tr = G_b_tr[np.sort(np.random.choice(len(G_b_tr), 6000, replace=False))]
    np.random.seed(42)
    if len(G_a_tr) > 3000:
        G_a_tr = G_a_tr[np.sort(np.random.choice(len(G_a_tr), 3000, replace=False))]

    best = {'f1': -1.0}

    def _objective(trial):
        nonlocal best
        w = trial.suggest_categorical('weight_rbf', list(weight_rbfs))
        w_lap = trial.suggest_categorical('weight_lap', [0.0, 0.1, 0.2, 0.3, 0.4])

        alpha_g, trainX_g, params_g = kfda_projection(
            G_b_tr, G_a_tr, weight_rbf=w, weight_lap=w_lap)

        S_calib = {fp: score_trace(fp, trainX_g, alpha_g, params_g,
                                   steer_scale, group_name) for fp in val_calib}
        S_eval = {fp: score_trace(fp, trainX_g, alpha_g, params_g,
                                  steer_scale, group_name) for fp in val_eval}
        S_atk = {fp: score_trace(fp, trainX_g, alpha_g, params_g,
                                 steer_scale, group_name) for fp in ATK_VAL}
        S_train_ref = {fp: score_trace(fp, trainX_g, alpha_g, params_g,
                                       steer_scale, group_name)
                       for fp in BENIGN_TRAIN}

        inner_best_f1, inner_best = -1.0, None
        for qd in q_deadzones:
            for bs in block_sizes:
                cp = calibrate_cusum_for_kfda(
                    S_calib, V_calib, v_ref, q_deadzone=qd, block_size=bs,
                    n_bootstrap=min(n_bootstrap, 1000), fpr_target=0.0005)

                def _calib_h(s_full, v_full):
                    ws_s = s_full[IGNORE_STARTUP:min(WARMUP[1], len(s_full))]
                    ws_v = v_full[IGNORE_STARTUP:min(WARMUP[1], len(v_full))]
                    if len(ws_s) <= 10:
                        return cp['h_up'], cp['h_down']
                    f = compute_scene_calibration_factor(
                        ws_s, V_train_ref, S_train_ref, ws_v, cp,
                        warmup_start=0, warmup_end=len(ws_s))
                    scale = float(np.clip(f, 1.0, 4.0))
                    return cp['h_up'] * scale, cp['h_down'] * scale

                tp = fp_cnt = fn = 0
                for fp in val_eval:
                    hu, hd = _calib_h(S_eval[fp], V_eval[fp])
                    a, sus, pk = cusum_alarm_drift_corrected(
                        S_eval[fp], V_eval[fp], cp['k_proj'], cp['B_UP'],
                        cp['B_DN'], hu, hd, CONSEC, cp['v_ref'],
                        return_gate_stats=True)
                    if (a > IGNORE_STARTUP and sus >= ELITE_SUSTAIN_MIN
                            and pk >= ELITE_PEAK_MIN):
                        fp_cnt += 1
                for fp in ATK_VAL:
                    atk = attack_start.get(fp, 402)
                    hu, hd = _calib_h(S_atk[fp], V_atk[fp])
                    a, sus, pk = cusum_alarm_drift_corrected(
                        S_atk[fp], V_atk[fp], cp['k_proj'], cp['B_UP'],
                        cp['B_DN'], hu, hd, CONSEC, cp['v_ref'],
                        return_gate_stats=True)
                    ok = (a >= 0 and sus >= ELITE_SUSTAIN_MIN
                          and pk >= ELITE_PEAK_MIN)
                    if not ok:
                        fn += 1
                    elif a < atk:               # pre-attack alarm counts as FP
                        fp_cnt += 1
                    else:
                        tp += 1
                prec = tp / (tp + fp_cnt) if (tp + fp_cnt) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                if f1 > inner_best_f1:
                    inner_best_f1 = f1
                    inner_best = {'w': w, 'w_lap': w_lap, 'qd': qd, 'bs': bs,
                                  'f1': f1, 'precision': prec, 'recall': rec}

        print(f"  trial#{trial.number:3d} w_rbf={w:.1f} w_lap={w_lap:.1f}: "
              f"best inner F1={inner_best_f1:.3f} "
              f"(qd={inner_best['qd']}, bs={inner_best['bs']})")
        if inner_best_f1 > best['f1']:
            best = dict(inner_best, alpha=alpha_g, trainX=trainX_g,
                        params=params_g)
        return inner_best_f1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  [BEST {group_name}] w_rbf={best['w']} w_lap={best['w_lap']} "
          f"qd={best['qd']} bs={best['bs']} val_F1={best['f1']:.3f}")
    return best


def train_one_group(group_name, BENIGN_TRAIN, BENIGN_VAL, ATK_TRAIN, ATK_VAL,
                    steer_scale, v_ref, attack_start, V_BENIGN_CACHE):
    """Full training for one pathway: hyper-parameter search, benign score
    cache, and final CUSUM calibration with the attack-informed center."""
    if len(ATK_TRAIN) < 3 or len(ATK_VAL) < 3:
        print(f"[SKIP] pathway '{group_name}': too few attack traces "
              f"(train={len(ATK_TRAIN)}, val={len(ATK_VAL)})")
        return {'group_name': group_name, 'kfda_model': None,
                's_benign': {}, 'cusum_params': None, 'best_hparams': None}

    # GNSS-position pathways carry the weakest signal: widen the search.
    if 'gps_xy' in group_name:
        w_rbfs = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
        n_trials, n_bootstrap = 30, 5000
        block_sizes = (80, 150, 200, 400, 800)
        fpr_target = 0.001
    else:
        w_rbfs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        n_trials, n_bootstrap = 20, 300
        block_sizes = (80, 150)
        fpr_target = 0.0005

    best = search_best_hyperparams_for_group(
        group_name, BENIGN_TRAIN, BENIGN_VAL, ATK_TRAIN, ATK_VAL,
        steer_scale, v_ref, attack_start,
        weight_rbfs=w_rbfs,
        q_deadzones=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 0.9),
        block_sizes=block_sizes, n_bootstrap=n_bootstrap, n_trials=n_trials)

    s_benign = {fp: score_trace(fp, best['trainX'], best['alpha'],
                                best['params'], steer_scale, group_name)
                for fp in BENIGN_TRAIN}

    # Attack scores set the CUSUM center at the benign/attack midpoint.
    s_attack_list = []
    for fp in ATK_TRAIN:
        s_atk = score_trace(fp, best['trainX'], best['alpha'],
                            best['params'], steer_scale, group_name)
        s_attack_list.append((s_atk, attack_start.get(fp, 402)))

    cusum = calibrate_cusum_for_kfda(
        s_benign, V_BENIGN_CACHE, v_ref, q_deadzone=best['qd'],
        block_size=best['bs'], n_bootstrap=n_bootstrap,
        fpr_target=fpr_target, S_ATTACK=s_attack_list or None)

    return {'group_name': group_name,
            'kfda_model': {'alpha': best['alpha'], 'trainX': best['trainX'],
                           'params': best['params'],
                           'feature_group': group_name},
            's_benign': s_benign,
            'cusum_params': cusum,
            'best_hparams': {'weight_rbf': best['w'], 'weight_lap': best['w_lap'],
                             'q_deadzone': best['qd'], 'block_size': best['bs'],
                             'val_f1': best['f1']}}


# ==========================================================================
# Dataset scanning and town-stratified split
# ==========================================================================
_TOWN_RE = re.compile(r'(Town\d+\w*)')


def _extract_town(dirname):
    m = _TOWN_RE.search(dirname)
    return m.group(1) if m else 'Unknown'


def auto_scan_dataset(dataset_dirs):
    """Scan each root for benign/ and attack/ trace folders. The attack
    onset frame is read from savior_log.csv (first attack_on==1 row)."""
    benign_files, attack_files, attack_start = [], [], {}
    seen_b, seen_a = set(), set()
    for root in dataset_dirs:
        if not os.path.isdir(root):
            print(f"[WARN] dataset root not found: {root}")
            continue
        bdir = os.path.join(root, "benign")
        if os.path.isdir(bdir):
            for d in sorted(os.listdir(bdir)):
                csv_path = os.path.join(bdir, d, "curvature-consistency.csv")
                log_path = os.path.join(bdir, d, "savior_log.csv")
                if os.path.exists(csv_path) and os.path.exists(log_path) \
                        and os.path.getsize(csv_path) > 100 and d not in seen_b:
                    seen_b.add(d)
                    benign_files.append(csv_path)
        adir = os.path.join(root, "attack")
        if os.path.isdir(adir):
            for d in sorted(os.listdir(adir)):
                csv_path = os.path.join(adir, d, "curvature-consistency.csv")
                log_path = os.path.join(adir, d, "savior_log.csv")
                if not (os.path.exists(csv_path) and os.path.exists(log_path)
                        and os.path.getsize(csv_path) > 100) or d in seen_a:
                    continue
                seen_a.add(d)
                atk_step = 402
                try:
                    df_log = pd.read_csv(log_path)
                    if 'attack_on' in df_log.columns:
                        first = df_log[df_log['attack_on'] == 1]
                        if len(first) > 0:
                            atk_step = int(round(float(first.iloc[0]['t']) / 0.05))
                except Exception as e:
                    print(f"[WARN] could not parse {log_path}: {e}")
                attack_files.append(csv_path)
                attack_start[csv_path] = atk_step
    print(f"[SCAN] benign={len(benign_files)}, attack={len(attack_files)}")
    return benign_files, attack_files, attack_start


def filter_attacks(attack_files):
    """Release scope: attack-* prefix; stealthy types must be whitelisted;
    non-stealthy (overt) types are always kept, test-only."""
    kept = []
    for fp in attack_files:
        dn = os.path.basename(os.path.dirname(fp))
        if not dn.startswith("attack-"):
            continue
        if "stealthy" in dn:
            if any(f"-{x}-" in dn for x in INCLUDE_ATTACKS):
                kept.append(fp)
        else:
            kept.append(fp)
    return kept


def town_stratified_split(files, ratios=(0.6, 0.2, 0.2), seed=42, shuffle=False):
    """Split each town bucket independently 60/20/20 so that train, val,
    and test all span every town (paper Section V-A)."""
    by_town = defaultdict(list)
    for fp in files:
        by_town[_extract_town(os.path.basename(os.path.dirname(fp)))].append(fp)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for town in sorted(by_town):
        fps = sorted(by_town[town])
        if shuffle:
            fps = [fps[i] for i in rng.permutation(len(fps))]
        n = len(fps)
        if n < 3:
            n_tr, n_va = n, 0
        else:
            n_tr = max(1, int(round(ratios[0] * n)))
            n_va = max(1, int(round(ratios[1] * n)))
        train += fps[:n_tr]
        val += fps[n_tr:n_tr + n_va]
        test += fps[n_tr + n_va:]
    return train, val, test


def build_splits(benign_files, attack_files):
    """Benign: town-stratified 60/20/20 (shuffled within each town).
    Attacks: per (attack_type, town) 60/20/20 into the owning pathway's
    train/val; types outside every pathway (stealthyS1 and all overt
    types) go entirely to test."""
    benign_train, benign_val, benign_test = town_stratified_split(
        benign_files, shuffle=True)

    grouped = defaultdict(list)
    for fp in attack_files:
        dn = os.path.basename(os.path.dirname(fp))
        parts = dn.split("-")
        grouped[parts[1] if len(parts) >= 2 else "unknown"].append(fp)

    train_by_group = {g: [] for g in KFDA_GROUPS}
    val_by_group = {g: [] for g in KFDA_GROUPS}
    test_attacks = []
    for group_name, types in ATTACK_GROUPS.items():
        for atk_type in sorted(types):
            if atk_type not in grouped:
                continue
            tr, va, te = town_stratified_split(grouped[atk_type])
            train_by_group[group_name] += tr
            val_by_group[group_name] += va
            test_attacks += te
    for atk_type in sorted(grouped):
        if atk_type not in ALL_TRAINABLE_ATTACKS:
            test_attacks += grouped[atk_type]

    for g in KFDA_GROUPS:
        assert not (set(train_by_group[g]) & set(val_by_group[g]))
        assert not (set(train_by_group[g]) & set(test_attacks))
    return (benign_train, benign_val, benign_test,
            train_by_group, val_by_group, test_attacks)


# ==========================================================================
# Training and evaluation pipelines
# ==========================================================================
def train(benign_train, benign_val, train_by_group, val_by_group, attack_start):
    steer_scale = calibrate_steering_ratio(benign_train)
    print(f"[TRAIN] steer_scale = {steer_scale:.4f}")

    all_v = np.concatenate([load_trace(fp)["v"].to_numpy()
                            for fp in benign_train])
    moving = all_v[np.abs(all_v) > 1.0]
    v_ref = float(np.percentile(moving, 75)) if len(moving) else 8.0
    print(f"[TRAIN] v_ref (P75 moving speed) = {v_ref:.2f} m/s")

    # Layer-1 thresholds from benign training data alone.
    eps_b = [compute_consistency(load_trace(fp), steer_scale)
             for fp in benign_train]
    info = calibrate_layer1_thresholds(eps_b, {'v_ref': v_ref,
                                               'steer_scale': steer_scale})
    info['kfda'] = {}

    V_BENIGN_CACHE = {fp: load_trace(fp)["v"].to_numpy()
                      for fp in benign_train}

    results = Parallel(n_jobs=2, prefer='threads')(
        delayed(train_one_group)(
            g, benign_train, benign_val, train_by_group[g], val_by_group[g],
            steer_scale, v_ref, attack_start, V_BENIGN_CACHE)
        for g in KFDA_GROUPS)

    KFDA_MODELS, S_BENIGN_CACHE, BEST_HPARAMS = {}, {}, {}
    for r in results:
        g = r['group_name']
        KFDA_MODELS[g] = r['kfda_model']
        S_BENIGN_CACHE[g] = r['s_benign']
        if r['cusum_params'] is not None:
            info['kfda'][g] = r['cusum_params']
        if r['best_hparams'] is not None:
            BEST_HPARAMS[g] = r['best_hparams']

    joblib.dump({'KFDA_MODELS': KFDA_MODELS, 'info': info,
                 'S_BENIGN_CACHE': S_BENIGN_CACHE,
                 'V_BENIGN_CACHE': V_BENIGN_CACHE,
                 'steer_scale': steer_scale,
                 'BEST_HPARAMS': BEST_HPARAMS}, CACHE_FILE)
    print(f"[TRAIN] saved: {CACHE_FILE}")


def evaluate(benign_test, test_attacks, attack_start):
    cache = joblib.load(CACHE_FILE)
    KFDA_MODELS = cache['KFDA_MODELS']
    info = cache['info']
    S_BENIGN_CACHE = cache['S_BENIGN_CACHE']
    V_BENIGN_CACHE = cache['V_BENIGN_CACHE']
    steer_scale = cache['steer_scale']
    BEST_HPARAMS = cache['BEST_HPARAMS']

    # A pathway is active iff its validation F1 clears the threshold.
    active = {g for g in KFDA_GROUPS
              if isinstance(BEST_HPARAMS.get(g), dict)
              and float(BEST_HPARAMS[g].get('val_f1', 0.0)) >= ACTIVE_F1_THRESHOLD}
    if not active:
        active = {g for g, m in KFDA_MODELS.items() if m is not None}
    print(f"[TEST] active pathways: {sorted(active)}")
    print(f"[TEST] benign={len(benign_test)}, attack={len(test_attacks)}")

    def _run(fp):
        df = load_trace(fp)
        if len(df) < 60:
            return None
        l1, l2, _ = ensemble_detector(
            df, KFDA_MODELS, info, steer_scale,
            S_BENIGN_CACHE, V_BENIGN_CACHE, active_groups=active)
        fired = [a for a in (l1, l2) if a != -1]
        return min(fired) if fired else -1

    # Metric convention of paper Eq. 10 (TOLERANCE = 0):
    #   P = TP / (TP + FP_pre + FP_ben),  R = TP / (TP + FN + FP_pre).
    TP = FP_pre = FP_ben = TN = FN = 0
    recall_by_type = defaultdict(lambda: {'total': 0, 'hit': 0})

    for fp in benign_test:
        alarm = _run(fp)
        if alarm is None:
            continue
        if alarm == -1:
            TN += 1
        else:
            FP_ben += 1
        print(f"  [benign] {os.path.basename(os.path.dirname(fp))[:40]}: "
              f"alarm={alarm} -> {'FP' if alarm != -1 else 'TN'}")

    for fp in test_attacks:
        alarm = _run(fp)
        if alarm is None:
            continue
        atk = attack_start.get(fp, 402)
        dn = os.path.basename(os.path.dirname(fp))
        atk_type = dn.split("-")[1] if len(dn.split("-")) >= 2 else "unknown"
        recall_by_type[atk_type]['total'] += 1
        if alarm == -1:
            FN += 1
            label = 'FN'
        elif alarm < atk:
            FP_pre += 1
            label = 'FP(pre-attack)'
        else:
            TP += 1
            recall_by_type[atk_type]['hit'] += 1
            label = f'TP (delay={alarm - atk})'
        print(f"  [attack] {dn[:40]} ({atk_type}): alarm={alarm}, "
              f"onset={atk} -> {label}")

    print("\n[RECALL BY ATTACK TYPE]")
    for at in sorted(recall_by_type):
        s = recall_by_type[at]
        r = s['hit'] / s['total'] if s['total'] else 0
        print(f"  {at:22s} {s['hit']:4d}/{s['total']:4d} = {r * 100:5.1f}%")

    prec = TP / max(TP + FP_pre + FP_ben, 1)
    rec = TP / max(TP + FN + FP_pre, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n[OVERALL]  TP={TP}  FP_pre={FP_pre}  FP_ben={FP_ben}  "
          f"TN={TN}  FN={FN}")
    print(f"[OVERALL]  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == "__main__":
    benign_files, attack_files, attack_start = auto_scan_dataset(DATASET_DIRS)
    attack_files = filter_attacks(attack_files)

    (benign_train, benign_val, benign_test,
     train_by_group, val_by_group, test_attacks) = build_splits(
        benign_files, attack_files)

    print("[SPLIT] benign: "
          f"train={len(benign_train)}, val={len(benign_val)}, "
          f"test={len(benign_test)}")
    for g in KFDA_GROUPS:
        print(f"[SPLIT] pathway {g}: train={len(train_by_group[g])}, "
              f"val={len(val_by_group[g])}")
    print(f"[SPLIT] test attacks: {len(test_attacks)}")

    if not os.path.exists(CACHE_FILE):
        train(benign_train, benign_val, train_by_group, val_by_group,
              attack_start)
    evaluate(benign_test, test_attacks, attack_start)
