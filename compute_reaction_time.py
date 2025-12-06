#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # for batch jobs (no display)
import matplotlib.pyplot as plt


def normalize(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-8:
        return x * 0.0
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def cross_correlation_rt(stim, gaze, fps,
                         min_rt_ms=100.0,
                         max_rt_ms=1500.0):
    """
    Compute RT via cross-correlation peak (stim -> gaze) within a plausible window.

    stim, gaze: 1D arrays (already aligned roughly in time)
    fps: frames per second
    min_rt_ms, max_rt_ms: search RT only in [min_rt_ms, max_rt_ms]

    Returns (rt_ms or None, lags, corr)
    """
    stim_n = normalize(stim)
    gaze_n = normalize(gaze)

    # corr[k] ~= sum_t stim_n[t] * gaze_n[t + lag]
    corr = np.correlate(gaze_n, stim_n, mode="full")
    lags = np.arange(-len(stim_n) + 1, len(gaze_n))

    # convert RT window to lag samples (gaze should LAG stimulus => lag >= 0)
    min_lag_samples = int(np.floor((min_rt_ms / 1000.0) * fps))
    max_lag_samples = int(np.ceil((max_rt_ms / 1000.0) * fps))

    # keep only non-negative lags within the window
    mask = (lags >= min_lag_samples) & (lags <= max_lag_samples)
    if not np.any(mask):
        return None, lags, corr

    corr_win = corr[mask]
    lags_win = lags[mask]

    k = int(np.argmax(corr_win))
    lag_samples = int(lags_win[k])

    rt_ms = (lag_samples / float(fps)) * 1000.0
    return float(rt_ms), lags, corr


def change_point_rt(stim, gaze, fps,
                    stim_thresh=0.5,
                    gaze_frac=0.3,
                    max_rt_ms=1500.0):
    """
    Simple change-point RT:

      - Find FIRST big change in stimulus: |Δstim| > stim_thresh (z-score units).
      - Then find FIRST change in gaze after that where |Δgaze| > gaze_frac * max|Δgaze|.
      - Restrict RT to [0, max_rt_ms]; otherwise return None.
    """
    s = normalize(stim)
    g = normalize(gaze)

    ds = np.diff(s, prepend=s[0])
    dg = np.diff(g, prepend=g[0])

    step_idx = np.where(np.abs(ds) > stim_thresh)[0]
    if len(step_idx) == 0:
        return None

    first = int(step_idx[0])

    max_dg = float(np.max(np.abs(dg)))
    if max_dg <= 0:
        return None

    thr = gaze_frac * max_dg

    # only consider gaze responses AFTER the stim step
    resp_candidates = np.where(
        (np.arange(len(dg)) > first) & (np.abs(dg) > thr)
    )[0]
    if len(resp_candidates) == 0:
        return None

    resp = int(resp_candidates[0])
    rt_ms = ((resp - first) / float(fps)) * 1000.0

    if rt_ms < 0 or rt_ms > max_rt_ms:
        return None

    return float(rt_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", required=True,
                    help="stimulus_trace.csv (from compute_stimulus_trace)")
    ap.add_argument("--gaze", required=True,
                    help="user_gaze.csv (from infer_user_video_vit)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for RT plots/summary")
    ap.add_argument("--fps", type=float, default=30.0)
    # Optional RT window overrides
    ap.add_argument("--min-rt-ms", type=float, default=100.0)
    ap.add_argument("--max-rt-ms", type=float, default=1500.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stim_df = pd.read_csv(args.stimulus)
    gaze_df = pd.read_csv(args.gaze)

    # Use stimulus cx and gaze yaw; we assume they are roughly aligned in time
    stim_signal = stim_df["cx"].to_numpy()
    gaze_signal = gaze_df["yaw_deg"].to_numpy()

    # Align lengths
    L = min(len(stim_signal), len(gaze_signal))
    stim_signal = stim_signal[:L]
    gaze_signal = gaze_signal[:L]

    # Cross-corr RT within plausible window
    rt_xcorr_ms, lags, corr = cross_correlation_rt(
        stim_signal,
        gaze_signal,
        fps=args.fps,
        min_rt_ms=args.min_rt_ms,
        max_rt_ms=args.max_rt_ms,
    )

    # Change-point RT within same max window
    rt_cp_ms = change_point_rt(
        stim_signal,
        gaze_signal,
        fps=args.fps,
        stim_thresh=0.5,
        gaze_frac=0.3,
        max_rt_ms=args.max_rt_ms,
    )

    print(f"[rt] RT cross-corr (ms)   : {rt_xcorr_ms}")
    print(f"[rt] RT change-point (ms): {rt_cp_ms}")

    # Save summary JSON
    summary_path = out_dir / "rt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "rt_xcorr_ms": rt_xcorr_ms,
                "rt_changepoint_ms": rt_cp_ms,
                "min_rt_ms": args.min_rt_ms,
                "max_rt_ms": args.max_rt_ms,
            },
            f,
            indent=2,
        )
    print(f"[rt] Saved summary to {summary_path}")

    # Cross-correlation plot (full lags)
    plt.figure(figsize=(8, 5))
    plt.plot(lags, corr)
    plt.title("Cross-correlation (stim cx vs gaze yaw)")
    plt.xlabel("lag (samples)")
    plt.ylabel("corr")
    plt.tight_layout()
    cc_path = out_dir / "rt_crosscorr.png"
    plt.savefig(cc_path)
    plt.close()
    print(f"[rt] Saved cross-corr plot to {cc_path}")

    # Overlay normalized stim vs gaze
    tt = np.arange(L) / args.fps
    ss = normalize(stim_signal)
    gg = normalize(gaze_signal)

    plt.figure(figsize=(10, 4))
    plt.plot(tt, ss, label="stim (cx)")
    plt.plot(tt, gg, label="gaze (yaw)")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("z-score")
    plt.title("Stimulus vs Gaze (normalized)")
    plt.tight_layout()
    ov_path = out_dir / "rt_overlay.png"
    plt.savefig(ov_path)
    plt.close()
    print(f"[rt] Saved overlay plot to {ov_path}")


if __name__ == "__main__":
    main()
