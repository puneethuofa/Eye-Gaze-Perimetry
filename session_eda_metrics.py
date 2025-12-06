#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-8:
        return x * 0.0
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def compute_basic_stats(stim_x, gaze_y):
    stats = {}
    stats["stim_mean"] = float(np.mean(stim_x))
    stats["stim_std"] = float(np.std(stim_x))
    stats["gaze_mean"] = float(np.mean(gaze_y))
    stats["gaze_std"] = float(np.std(gaze_y))

    r = np.corrcoef(stim_x, gaze_y)[0, 1]
    r_neg = np.corrcoef(stim_x, -gaze_y)[0, 1]

    stats["pearson_r_yaw"] = float(r)
    stats["pearson_r_neg_yaw"] = float(r_neg)

    # choose sign with stronger magnitude
    if abs(r_neg) > abs(r):
        stats["best_sign"] = -1
        stats["best_r"] = float(r_neg)
    else:
        stats["best_sign"] = 1
        stats["best_r"] = float(r)
    return stats


def cross_correlation(stim, gaze):
    stim_n = normalize(stim)
    gaze_n = normalize(gaze)
    corr = np.correlate(gaze_n, stim_n, mode="full")
    lags = np.arange(-len(stim_n) + 1, len(gaze_n))
    return lags, corr


def detect_stim_jumps(stim, fps, z_thresh=0.7, min_separation_s=0.5):
    """
    Detect approximate stimulus 'jumps' using derivative in z-score units.
    Returns array of indices where large jumps occur, spaced by min_separation_s.
    """
    s = normalize(stim)
    ds = np.diff(s, prepend=s[0])
    cand = np.where(np.abs(ds) > z_thresh)[0]
    if len(cand) == 0:
        return np.array([], dtype=int)

    min_sep = int(round(min_separation_s * fps))
    events = [int(cand[0])]
    last = cand[0]
    for idx in cand[1:]:
        if idx - last >= min_sep:
            events.append(int(idx))
            last = idx
    return np.array(events, dtype=int)


def per_event_rt(stim, gaze, t_sec, fps,
                 stim_events,
                 gaze_frac=0.3,
                 max_rt_ms=1500.0):
    """
    For each stimulus event index, find a gaze response in a window after the event
    using a derivative threshold on gaze.

    Returns DataFrame with:
        event_idx, event_time, stim_delta, rt_ms (NaN if no clear response)
    """
    s = normalize(stim)
    g = normalize(gaze)
    dg = np.diff(g, prepend=g[0])

    max_abs_dg = float(np.max(np.abs(dg))) if len(dg) > 0 else 0.0
    rows = []
    max_rt_samples = int(round((max_rt_ms / 1000.0) * fps))

    for idx in stim_events:
        if idx >= len(s) - 1:
            continue
        t0 = float(t_sec[idx])
        stim_delta = float(s[idx + 1] - s[idx])

        start = idx + 1
        end = min(len(dg), idx + 1 + max_rt_samples)
        if start >= end or max_abs_dg <= 0:
            rt_ms = np.nan
        else:
            local_dg = dg[start:end]
            thr = gaze_frac * max_abs_dg
            cand = np.where(np.abs(local_dg) > thr)[0]
            if len(cand) == 0:
                rt_ms = np.nan
            else:
                resp_idx = start + int(cand[0])
                rt_ms = ((resp_idx - idx) / float(fps)) * 1000.0

        rows.append(
            dict(
                event_idx=int(idx),
                event_time=t0,
                stim_delta=stim_delta,
                rt_ms=rt_ms,
            )
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", required=True, help="stimulus_trace.csv path")
    ap.add_argument("--gaze", required=True, help="user_gaze.csv path")
    ap.add_argument("--out-dir", required=True, help="directory for EDA outputs")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stim_df = pd.read_csv(args.stimulus)
    gaze_df = pd.read_csv(args.gaze)

    # Align by length
    L = min(len(stim_df), len(gaze_df))
    stim_df = stim_df.iloc[:L].reset_index(drop=True)
    gaze_df = gaze_df.iloc[:L].reset_index(drop=True)

    t = stim_df["t_sec"].to_numpy()
    stim_x = stim_df["cx"].to_numpy()
    yaw = gaze_df["yaw_deg"].to_numpy()
    pitch = gaze_df["pitch_deg"].to_numpy()

    # --- 1. Basic stats + correlations ---
    basic_stats = compute_basic_stats(stim_x, yaw)
    basic_stats["n_samples"] = int(L)
    basic_stats_path = out_dir / "eda_basic_stats.json"
    with open(basic_stats_path, "w") as f:
        json.dump(basic_stats, f, indent=2)

    # --- 2. Cross-correlation (yaw and -yaw) ---
    lags, corr = cross_correlation(stim_x, yaw)
    _, corr_neg = cross_correlation(stim_x, -yaw)

    plt.figure(figsize=(8, 5))
    plt.plot(lags, corr, label="yaw")
    plt.plot(lags, corr_neg, label="-yaw", linestyle="--")
    plt.axhline(0, color="k", linewidth=0.5)
    plt.xlabel("lag (samples)")
    plt.ylabel("corr")
    plt.title("Cross-correlation (stim_x vs yaw / -yaw)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eda_crosscorr_yaw.png")
    plt.close()

    # --- 3. Time-series overlay (normalized, best sign) ---
    s_norm = normalize(stim_x)
    signed_yaw = yaw * basic_stats["best_sign"]
    y_norm = normalize(signed_yaw)

    plt.figure(figsize=(10, 4))
    plt.plot(t, s_norm, label="stim_x (z)")
    plt.plot(t, y_norm, label=f"{'yaw' if basic_stats['best_sign']==1 else '-yaw'} (z)")
    plt.xlabel("time (s)")
    plt.ylabel("z-score")
    plt.title("Stimulus X vs Gaze Yaw (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eda_ts_overlay.png")
    plt.close()

    # --- 4. Scatter plots ---
    plt.figure(figsize=(5, 5))
    plt.scatter(stim_x, yaw, s=4, alpha=0.5)
    plt.xlabel("stim_x (pixels)")
    plt.ylabel("yaw (deg)")
    plt.title("Scatter: stim_x vs yaw")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_scatter_stimx_yaw.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(stim_x, signed_yaw, s=4, alpha=0.5)
    plt.xlabel("stim_x (pixels)")
    plt.ylabel(f"{'yaw' if basic_stats['best_sign']==1 else '-yaw'} (deg)")
    plt.title("Scatter: stim_x vs signed yaw")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_scatter_stimx_signedyaw.png")
    plt.close()

    # --- 5. Histograms of yaw and pitch ---
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(yaw, bins=40)
    plt.xlabel("yaw (deg)")
    plt.ylabel("count")
    plt.title("Yaw distribution")

    plt.subplot(1, 2, 2)
    plt.hist(pitch, bins=40)
    plt.xlabel("pitch (deg)")
    plt.title("Pitch distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_hist_yaw_pitch.png")
    plt.close()

    # --- 6. Per-event RTs (per stimulus jump) ---
    stim_events = detect_stim_jumps(
        stim_x, fps=args.fps, z_thresh=0.7, min_separation_s=0.6
    )
    rt_df = per_event_rt(
        stim_x,
        signed_yaw,
        t,
        fps=args.fps,
        stim_events=stim_events,
        gaze_frac=0.3,
        max_rt_ms=1500.0,
    )
    rt_df.to_csv(out_dir / "eda_per_event_rt.csv", index=False)

    # RT histogram
    valid_rts = rt_df["rt_ms"].replace({np.nan: None}).dropna().to_numpy()
    if len(valid_rts) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(valid_rts, bins=20)
        plt.xlabel("RT (ms)")
        plt.ylabel("count")
        plt.title("Per-event RT distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "eda_hist_rt_per_event.png")
        plt.close()

    # Per-event RT summary
    rt_summary = {
        "n_events_detected": int(len(rt_df)),
        "n_events_with_rt": int(np.sum(~np.isnan(rt_df["rt_ms"]))),
    }
    if len(valid_rts) > 0:
        rt_summary.update(
            {
                "rt_ms_mean": float(np.mean(valid_rts)),
                "rt_ms_median": float(np.median(valid_rts)),
                "rt_ms_std": float(np.std(valid_rts)),
                "rt_ms_min": float(np.min(valid_rts)),
                "rt_ms_max": float(np.max(valid_rts)),
            }
        )
    summary_path = out_dir / "eda_per_event_rt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rt_summary, f, indent=2)


if __name__ == "__main__":
    main()

