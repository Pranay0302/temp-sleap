#!/usr/bin/env python3
from pathlib import Path
import csv
import argparse
import math
import sys

# Optional pretty progress bar
try:
    from tqdm import tqdm  # pip install tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# Prefer PyAV for MKV robustness
try:
    import av  # pip install av
    HAVE_PYAV = True
except Exception:
    HAVE_PYAV = False

import cv2
import numpy as np


def ensure_even(w, h):
    return (w - (w % 2), h - (h % 2))


def estimate_total_frames_from_stream(stream):
    # 1) try stream.frames
    if getattr(stream, "frames", 0):
        return int(stream.frames)
    # 2) try duration * fps
    try:
        if stream.duration and stream.time_base and stream.average_rate:
            dur = float(stream.duration * stream.time_base)  # seconds
            fps = float(stream.average_rate)
            return int(round(dur * fps))
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Rewrite any video to AVI/MJPG (constant FPS) for SLEAP.")
    ap.add_argument("input", help="Path to the source video (e.g., .mkv)")
    ap.add_argument("-o", "--out", help="Output .avi path (default: <stem>_mjpg.avi next to input)")
    ap.add_argument("--fps", type=float, default=None, help="Force FPS (e.g., 30). If omitted, auto-detect.")
    ap.add_argument("--quality", type=int, default=90, help="MJPG quality 1-100 (default 90).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_suffix("").parent / f"{in_path.stem}_mjpg.avi"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"Output already exists: {out_path}\nUse --overwrite to replace.")

    ts_csv = out_path.with_suffix(".timestamps.csv")

    if HAVE_PYAV:
        print("Backend: PyAV")
        container = av.open(str(in_path))
        vstream = next(s for s in container.streams if s.type == "video")
        vstream.thread_type = "AUTO"

        # FPS
        if args.fps:
            fps = float(args.fps)
        else:
            # average_rate is the most reliable
            if vstream.average_rate:
                fps = float(vstream.average_rate)
            else:
                # Fallback: derive from first two frames
                gen = container.decode(vstream)
                f1 = next(gen)
                f2 = next(gen)
                tb = float(vstream.time_base) if vstream.time_base else 0.0
                fps = round(1.0 / ((f2.pts - f1.pts) * tb)) if tb and f1.pts is not None and f2.pts is not None else 30.0
                container.close()
                container = av.open(str(in_path))
                vstream = next(s for s in container.streams if s.type == "video")
                vstream.thread_type = "AUTO"

        # Frame size
        width, height = vstream.codec_context.width, vstream.codec_context.height
        width, height = ensure_even(width, height)

        total = estimate_total_frames_from_stream(vstream)

        print(f"Input: {in_path}")
        if total:
            print(f"Estimated frames: {total}")
        print(f"Size: {width}x{height}  |  Target FPS: {fps:.3f}")
        print(f"Output: {out_path}")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            sys.exit("ERROR: Could not open VideoWriter for output.")

        # Some backends honor this:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(args.quality))

        count = 0
        bar = tqdm(total=total, unit="f") if HAVE_TQDM and total else None

        try:
            with open(ts_csv, "w", newline="") as fcsv:
                wcsv = csv.writer(fcsv)
                wcsv.writerow(["frame_idx", "source_pts", "source_time_seconds"])
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="bgr24")
                    if (img.shape[1], img.shape[0]) != (width, height):
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(img)

                    t = None
                    if vstream.time_base and frame.pts is not None:
                        t = float(frame.pts * vstream.time_base)
                    wcsv.writerow([count, frame.pts if frame.pts is not None else "", t if t is not None else ""])

                    count += 1
                    if bar:
                        bar.update(1)
                    elif count % 500 == 0:
                        print(f"... {count} frames")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if bar:
                bar.close()
            writer.release()
            container.close()

        print(f"Done. Wrote {count} frames to {out_path}")
        print(f"Timestamps: {ts_csv}")
        return

    # ---------------- OpenCV fallback ----------------
    print("Backend: OpenCV (PyAV not available). For MKV reliability, install PyAV:  pip install av")
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        sys.exit("ERROR: Could not open input with OpenCV either.")

    fps = float(args.fps) if args.fps else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = ensure_even(width, height)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

    print(f"Input: {in_path}")
    if total:
        print(f"Estimated frames: {total}")
    print(f"Size: {width}x{height}  |  Target FPS: {fps:.3f}")
    print(f"Output: {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit("ERROR: Could not open VideoWriter for output.")

    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(args.quality))

    count = 0
    bar = tqdm(total=total, unit="f") if HAVE_TQDM and total else None

    try:
        with open(ts_csv, "w", newline="") as fcsv:
            wcsv = csv.writer(fcsv)
            wcsv.writerow(["frame_idx"])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                wcsv.writerow([count])
                count += 1
                if bar:
                    bar.update(1)
                elif count % 500 == 0:
                    print(f"... {count} frames")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if bar:
            bar.close()
        cap.release()
        writer.release()

    print(f"Done. Wrote {count} frames to {out_path}")
    print(f"Timestamps: {ts_csv}")


if __name__ == "__main__":
    main()
