#!/usr/bin/env python3
import argparse
import csv
import sys
import cv2


def extract_stimulus_trace(
    video_path,
    out_csv_path,
    fps=30.0,
    frame_step=2,
    min_area=40.0,
    target_width=320,
    max_frames=None,
):
    """
    Ultra-lean stimulus tracker:

    - Asks OpenCV to decode frames at low resolution (target_width hint).
    - For every Nth frame (frame_step):
        * Convert to gray.
        * Simple fixed threshold for bright dot.
        * Largest contour -> centroid (cx, cy).
        * Write (t_sec, cx, cy) directly to CSV.
    - No frame buffering, no heavy operations.
    """

    # Reduce OpenCV threading to avoid extra memory fragmentation
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[stim] ERROR: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Hint to decoder to keep resolution low
    if target_width and target_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_width)

    step = max(1, int(frame_step))

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "cx", "cy"])

        frame_idx = 0       # actual frame index
        processed = 0       # frames checked
        written = 0         # rows written

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # safety upper bound
            if max_frames is not None and processed >= max_frames:
                break

            # only process every "step" frames
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            processed += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # simple fixed threshold, adjust 200 if needed
            _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            cnts, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not cnts:
                frame_idx += 1
                continue

            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < float(min_area):
                frame_idx += 1
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                frame_idx += 1
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            t = frame_idx * (1.0 / float(fps))
            writer.writerow([f"{t:.6f}", f"{cx:.3f}", f"{cy:.3f}"])
            written += 1

            frame_idx += 1

    cap.release()
    print(
        f"[stim] done. frames_seen={frame_idx}, processed={processed}, "
        f"rows_written={written}, out='{out_csv_path}'"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="stimulus video path")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=40.0)
    ap.add_argument("--target-width", type=int, default=320)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="optional hard limit on number of frames processed")
    args = ap.parse_args()

    extract_stimulus_trace(
        video_path=args.video,
        out_csv_path=args.out,
        fps=args.fps,
        frame_step=args.frame_step,
        min_area=args.min_area,
        target_width=args.target_width,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
