#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import cv2


def extract_stimulus_trace(
    video_path,
    out_csv_path,
    fps=30.0,
    frame_step=2,
    min_area=20.0,
    target_width=320,
    max_frames=None,
):
    """
    Stimulus tracker that uses background subtraction:

      - Read first frame as background (bg_gray).
      - For each subsequent frame:
          * convert to gray and optionally downscale.
          * diff = |gray - bg_gray|.
          * Otsu threshold on diff to get moving regions.
          * largest contour -> centroid (cx, cy).
          * write (t_sec, cx, cy) to CSV.
    """

    video_path = str(video_path)
    out_csv_path = str(out_csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[stim] ERROR: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # --- read first frame as background ---
    ok, bg = cap.read()
    if not ok:
        print(f"[stim] ERROR: cannot read first frame of {video_path}", file=sys.stderr)
        sys.exit(1)

    # optional downscale for speed
    if target_width and target_width > 0 and bg.shape[1] > target_width:
        scale = float(target_width) / float(bg.shape[1])
        bg = cv2.resize(
            bg,
            (int(bg.shape[1] * scale), int(bg.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    step = max(1, int(frame_step))

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "cx", "cy"])

        frame_idx = 0       # includes background frame
        processed = 0
        written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # hard cap
            if max_frames is not None and processed >= max_frames:
                break

            frame_idx += 1  # first "moving" frame is index 1

            if frame_idx % step != 0:
                continue

            processed += 1

            if target_width and target_width > 0 and frame.shape[1] > target_width:
                scale = float(target_width) / float(frame.shape[1])
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- background subtraction ---
            diff = cv2.absdiff(gray, bg_gray)

            # Otsu threshold on difference to get moving spot
            _, th = cv2.threshold(
                diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            cnts, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < float(min_area):
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            t = frame_idx * (1.0 / float(fps))
            writer.writerow([f"{t:.6f}", f"{cx:.3f}", f"{cy:.3f}"])
            written += 1

    cap.release()
    print(
        f"[stim] done. frames_seen={frame_idx}, processed={processed}, "
        f"rows_written={written}, out='{out_csv_path}'"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--target-width", type=int, default=320)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    extract_stimulus_trace(
        video_path=Path(args.video),
        out_csv_path=Path(args.out),
        fps=args.fps,
        frame_step=args.frame_step,
        min_area=args.min_area,
        target_width=args.target_width,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
