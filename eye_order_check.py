#!/usr/bin/env python3
"""Determine which physical camera is the LEFT eye, by measuring parallax.

Run this on the Pi (it needs the stereo venv for cv2), whenever the rig has been
re-seated or a depth result looks wrong:

    ~/stereo-mcp/venv/bin/python eye_order_check.py

Why this exists
---------------
`calibrate_stereo`'s built-in eye-order inference is unreliable on this rig - it
reports camera 0 and flags itself INCONCLUSIVE. This script settles the question
independently, and its answer (camera 1 is the left eye) has been confirmed by
two further cross-checks: the measured dy matches the calibrator's vertical
offset, and the far-object dx matches its x-offset-at-infinity.

The method
----------
A near object and a far object are template-matched from cam0 into cam1. Only
the DIFFERENCE between their horizontal shifts is meaningful: this rig has a
large constant offset in both axes even at infinity, so a single absolute dx
says nothing (that was the original "constant offset" bug - two different scenes
both reporting exactly 143.3 px of "disparity").

    dx := x_cam1 - x_cam0

Disparity grows as objects get nearer. If the NEAR object has the more positive
dx, then cam1 is behaving as the left eye; otherwise cam0 is.

Choosing regions
----------------
Pass ROIs as x0,x1,y0,y1 in full-resolution pixels (1640x1232). Pick things with
crisp VERTICAL edges - a plug, a book spine, a frame corner. Avoid glossy
surfaces (specular highlights move between viewpoints) and repeating patterns.
The near object should be well under a metre, the far one several metres or
more; the bigger the depth difference, the larger the margin.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

import cv2
import numpy as np

STREAM_URL = "http://127.0.0.1:8001"
MIN_CONFIDENCE = 0.6


def grab(cam: int, stream_url: str) -> np.ndarray:
    """Pull one greyscale frame per camera from the MJPEG streamer.

    The streamer owns both sensors continuously, so rpicam-still cannot open
    them - taking the frame from the streamer is the only option while the
    service is up, and it is also far faster.
    """
    url = f"{stream_url}/snapshot/cam{cam}.jpg"
    data = urllib.request.urlopen(url, timeout=10).read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"camera {cam}: frame could not be decoded")
    return img


def shift_of(label: str, cam0: np.ndarray, cam1: np.ndarray, roi) -> tuple[float, float, float]:
    """Template-match one ROI taken from cam0 into cam1. Returns (dx, dy, confidence)."""
    x0, x1, y0, y1 = roi
    h, w = cam0.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise SystemExit(f"{label}: ROI {roi} lies outside the {w}x{h} frame")

    template = cam0[y0:y1, x0:x1]
    result = cv2.matchTemplate(cam1, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(result)
    dx = float(location[0] - x0)
    dy = float(location[1] - y0)
    print(f"{label:<14} ROI x={x0}..{x1} y={y0}..{y1}"
          f"  ->  dx={dx:+7.1f} px  dy={dy:+6.1f} px  confidence={confidence:.3f}")
    return dx, dy, confidence


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--near", default="260,580,300,660",
                   help="ROI of a NEAR object as x0,x1,y0,y1 (default: the plug used on 2026-09-03)")
    p.add_argument("--far", default="1100,1310,295,530",
                   help="ROI of a FAR object as x0,x1,y0,y1 (default: the picture frame)")
    p.add_argument("--stream-url", default=STREAM_URL)
    args = p.parse_args()

    def parse_roi(s: str):
        try:
            parts = tuple(int(v) for v in s.split(","))
        except ValueError:
            raise SystemExit(f"could not parse ROI {s!r} - expected x0,x1,y0,y1")
        if len(parts) != 4:
            raise SystemExit(f"ROI {s!r} needs exactly four numbers: x0,x1,y0,y1")
        return parts

    cam0, cam1 = grab(0, args.stream_url), grab(1, args.stream_url)
    print(f"frames: cam0 {cam0.shape[1]}x{cam0.shape[0]}, cam1 {cam1.shape[1]}x{cam1.shape[0]}")
    print("template taken from cam0 and searched in cam1; dx = x_cam1 - x_cam0\n")

    near_dx, near_dy, near_conf = shift_of("NEAR", cam0, cam1, parse_roi(args.near))
    far_dx, far_dy, far_conf = shift_of("FAR", cam0, cam1, parse_roi(args.far))

    print(f"\nrelative disparity (near - far) = {near_dx - far_dx:+.1f} px")

    if min(near_conf, far_conf) < MIN_CONFIDENCE:
        print(f"\nINCONCLUSIVE - match confidence below {MIN_CONFIDENCE}. Re-aim at a "
              f"scene with crisper vertical edges, or pick different regions.")
        return 1

    left = 1 if near_dx > far_dx else 0
    print(f"\n==> LEFT eye is camera {left}, right eye is camera {1 - left}")
    print(f"    Pass this to the MCP tool explicitly: calibrate_stereo(left_camera={left})")

    # The two dy values describe the same rigid vertical misalignment, so they
    # should agree closely. If they do not, one of the matches is on the wrong
    # thing and the dx values are not trustworthy either.
    if abs(near_dy - far_dy) > 5:
        print(f"\n    WARNING: the two vertical shifts disagree "
              f"({near_dy:+.1f} vs {far_dy:+.1f} px). Both regions should show the "
              f"same rigid vertical offset, so at least one match is probably wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
