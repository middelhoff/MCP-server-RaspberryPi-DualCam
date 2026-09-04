#!/usr/bin/env python3
"""MCP server exposing the dual IMX219 stereo camera on a Raspberry Pi 5.

Serves over streamable HTTP so a client on the LAN (e.g. LM Studio) can call
the tools. Images are returned as MCP image content, downscaled by default so
a vision model does not get a 3280x2464 frame it has to burn tokens on.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import time
import urllib.request
from typing import Literal

import cv2
from PIL import Image as PILImage
from mcp.server.mcpserver import MCPServer, Image
from mcp.server.transport_security import TransportSecuritySettings

import vision

HOST = os.environ.get("STEREO_MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("STEREO_MCP_PORT", "8000"))

# Host header values accepted by the DNS-rebinding guard. Clients on the LAN
# send the Pi's address here, so it has to be allowed explicitly.
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get(
        "STEREO_MCP_ALLOWED_HOSTS",
        "192.168.178.35,raspberrypi,raspberrypi.local,localhost,127.0.0.1",
    ).split(",")
    if h.strip()
]

RPICAM = shutil.which("rpicam-still") or "/usr/bin/rpicam-still"

# tmpfs: capture scratch never touches the SD card
SCRATCH = "/dev/shm/stereo-mcp"

# Sensor native is 3280x2464; 1640x1232 is the binned mode and plenty for a VLM.
SENSOR_W, SENSOR_H = 1640, 1232

# Cameras, as reported by `rpicam-hello --list-cameras`.
CAM_LEFT = 1   # CAM1 port
CAM_RIGHT = 0  # CAM0 port
BASELINE_MM = 60.0

mcp = MCPServer(
    "raspi-stereo-camera",
    instructions=(
        "Live dual-IMX219 stereo camera on a Raspberry Pi 5, 60 mm baseline, "
        "with on-device analysis.\n\n"
        "If you CANNOT see images, use describe_scene (what is there and how far "
        "away) and depth_grid (distances across the view). These do the seeing "
        "on the Pi and return plain text.\n\n"
        "If you CAN see images, capture and capture_stereo_pair return photos "
        "directly.\n\n"
        "Depth requires calibration against a scene with varied, non-repeating "
        "texture. If a depth tool refuses, say so plainly and ask for the camera "
        "to be pointed at ordinary clutter, then call calibrate_stereo. Never "
        "guess a distance the tools declined to give."
        " There is also an ultrasonic distance sensor "
        "(ultraschall_abstand, weg_frei): it needs no calibration and works "
        "in the dark, but reports only the single nearest distance in a roughly "
        "15 degree cone straight ahead, with no direction and no object identity. "
        "Use it for a quick clearance check; use describe_scene or depth_grid "
        "when the question is what is there or where."
    ),
)


# The MJPEG streamer, when running, holds both sensors open - rpicam-still
# cannot then open them. So prefer fetching a frame from the streamer, and only
# drive the camera directly when the streamer is not up.
STREAM_URL = os.environ.get("STEREO_STREAM_URL", "http://127.0.0.1:8001")


def _snapshot_from_stream(cam: int, dest: str) -> bool:
    """Try to pull a full-res frame from the streamer. False if it isn't there."""
    try:
        with urllib.request.urlopen(
            f"{STREAM_URL}/snapshot/cam{cam}.jpg", timeout=5
        ) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
        if len(data) < 1000:
            return False
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except Exception:  # noqa: BLE001 - streamer absent or busy; fall back
        return False


def _capture_raw(cam: int, settle_ms: int, dest: str) -> None:
    """Run rpicam-still for one sensor. Raises on failure."""
    if _snapshot_from_stream(cam, dest):
        return
    cmd = [
        RPICAM, "-n",
        "--camera", str(cam),
        "-t", str(settle_ms),
        "--width", str(SENSOR_W),
        "--height", str(SENSOR_H),
        "-o", dest,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=settle_ms / 1000 + 20)
    if proc.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError(
            f"rpicam-still failed for camera {cam} (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[-400:]}"
        )


def _capture_both(settle_ms: int, dest_left: str, dest_right: str) -> None:
    """Fire both sensors concurrently, as close to simultaneous as software allows."""
    # If the streamer owns the cameras, take both frames from it. Its two
    # capture threads run free, so the pair is no less simultaneous than
    # launching two rpicam-still processes.
    got_a = _snapshot_from_stream(CAM_LEFT, dest_left)
    got_b = _snapshot_from_stream(CAM_RIGHT, dest_right)
    if got_a and got_b:
        return

    procs = []
    for cam, dest in ((CAM_LEFT, dest_left), (CAM_RIGHT, dest_right)):
        cmd = [
            RPICAM, "-n",
            "--camera", str(cam),
            "-t", str(settle_ms),
            "--width", str(SENSOR_W),
            "--height", str(SENSOR_H),
            "-o", dest,
        ]
        procs.append((cam, dest, subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.PIPE)))
    errors = []
    for cam, dest, proc in procs:
        _, stderr = proc.communicate(timeout=settle_ms / 1000 + 20)
        if proc.returncode != 0 or not os.path.exists(dest):
            errors.append(f"camera {cam}: {stderr.decode(errors='replace')[-300:]}")
    if errors:
        raise RuntimeError("simultaneous capture failed -> " + " | ".join(errors))


def _encode(path: str, max_width: int, quality: int = 85) -> bytes:
    """Downscale to max_width and re-encode as JPEG."""
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        if max_width and im.width > max_width:
            ratio = max_width / im.width
            im = im.resize((max_width, round(im.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def _side_by_side(left_path: str, right_path: str, max_width: int) -> bytes:
    """Stitch left|right into one frame, with a divider so the seam is obvious."""
    with PILImage.open(left_path) as li, PILImage.open(right_path) as ri:
        li, ri = li.convert("RGB"), ri.convert("RGB")
        half = max(1, max_width // 2)
        lw = half
        lh = round(li.height * (half / li.width))
        li = li.resize((lw, lh), PILImage.LANCZOS)
        ri = ri.resize((lw, round(ri.height * (half / ri.width))), PILImage.LANCZOS)
        gap = 6
        canvas = PILImage.new("RGB", (lw * 2 + gap, max(li.height, ri.height)), (20, 20, 20))
        canvas.paste(li, (0, 0))
        canvas.paste(ri, (lw + gap, 0))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


def _scratch(name: str) -> str:
    os.makedirs(SCRATCH, exist_ok=True)
    return os.path.join(SCRATCH, name)


@mcp.tool()
def camera_info() -> str:
    """Report the stereo rig's configuration and whether both sensors are live.

    Call this first if a capture fails, to see whether the cameras are detected
    at all.
    """
    try:
        out = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True, timeout=20,
        ).stdout.decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        out = f"(failed to list cameras: {exc})"

    detected = [ln.strip() for ln in out.splitlines() if ln.strip()[:2] in ("0 ", "1 ")]

    cal = vision.load_calibration()
    if not cal:
        cal_txt = ("NOT CALIBRATED. Depth tools will refuse until calibrate_stereo "
                   "has run against a suitable scene.")
    elif not cal.get("trustworthy", False):
        cal_txt = (f"Calibrated {cal.get('measured_at', '?')} but marked UNTRUSTWORTHY - "
                   "the scene it was measured against could not support stereo "
                   "matching. Depth tools will refuse. Re-run calibrate_stereo "
                   "pointed at ordinary clutter.")
    else:
        cal_txt = (f"Calibrated {cal['measured_at']}: left eye is camera "
                   f"{cal['left_camera']}, vertical offset {cal['y_shift_px']:+.1f} px, "
                   f"residual {cal['residual_px']:.1f} px. Depth available.")

    return (
        f"Raspberry Pi 5 stereo rig: dual Sony IMX219 8MP, {BASELINE_MM:.0f} mm baseline.\n"
        f"Capture resolution: {SENSOR_W}x{SENSOR_H} (binned mode)\n"
        f"Sensors detected: {len(detected)}\n"
        + ("\n".join(detected) if detected else "NONE - cameras are not enumerating")
        + f"\n\nCalibration: {cal_txt}\n"
        + "\nNote: there is no hardware frame sync between the two sensors. "
        "Stereo pairs are captured concurrently in software, so fast-moving "
        "subjects may not correspond exactly between the two frames."
    )


@mcp.tool()
def capture(
    eye: Literal["left", "right"] = "left",
    max_width: int = 1024,
    settle_ms: int = 1200,
) -> list:
    """Capture a single still from one eye of the stereo camera.

    Args:
        eye: Which sensor to use. "left" is CAM1, "right" is CAM0.
        max_width: Downscale the returned image to at most this width in pixels.
            Lower means fewer vision tokens. 1024 is a good default; use 1640
            for full binned resolution.
        settle_ms: Milliseconds to let auto-exposure and auto-white-balance
            converge before the frame is taken. Raise it in dim or changing light.
    """
    cam = CAM_LEFT if eye == "left" else CAM_RIGHT
    path = _scratch(f"single_{eye}.jpg")
    started = time.time()
    _capture_raw(cam, settle_ms, path)
    jpeg = _encode(path, max_width)
    elapsed = time.time() - started
    return [
        f"{eye.capitalize()} eye (camera {cam}), {len(jpeg) // 1024} KB, "
        f"captured in {elapsed:.1f}s.",
        Image(data=jpeg, format="jpeg"),
    ]


@mcp.tool()
def capture_stereo_pair(
    layout: Literal["separate", "side_by_side"] = "separate",
    max_width: int = 1024,
    settle_ms: int = 1200,
) -> list:
    """Capture both eyes at once and return them for stereo analysis.

    Use this for depth reasoning, disparity estimation, or any question about
    how far away something is. The two frames are taken concurrently.

    Args:
        layout: "separate" returns two images (left first, then right).
            "side_by_side" returns a single stitched image with left on the
            left half and right on the right half, which is often easier to
            compare in one glance.
        max_width: Width budget for the returned image(s), in pixels. For
            "side_by_side" this is the width of the whole stitched frame.
        settle_ms: Auto-exposure settle time in milliseconds before capture.
    """
    lpath, rpath = _scratch("stereo_left.jpg"), _scratch("stereo_right.jpg")
    started = time.time()
    _capture_both(settle_ms, lpath, rpath)
    elapsed = time.time() - started

    # Which sensor is physically the left eye is a calibration result. Until
    # that is established, say so rather than asserting an order the images may
    # contradict - a viewer reasoning about depth needs to know which is which.
    cal = vision.load_calibration()
    if cal.get("trustworthy"):
        first, second = cal["left_camera"], cal["right_camera"]
        eye_note = (f"The first image is the LEFT eye (camera {first}), the second "
                    f"is the RIGHT eye (camera {second}); this was measured, not assumed.")
    else:
        first, second = CAM_LEFT, CAM_RIGHT
        eye_note = (f"Images are in camera-index order ({first} then {second}). "
                    f"WHICH IS THE PHYSICAL LEFT EYE IS NOT YET ESTABLISHED - the rig "
                    f"has not been calibrated against a suitable scene. Do not draw "
                    f"conclusions that depend on left/right order; run calibrate_stereo "
                    f"against a cluttered scene first.")

    header = (
        f"Stereo pair from a {BASELINE_MM:.0f} mm baseline rig, captured "
        f"concurrently in {elapsed:.1f}s. Objects nearer the camera shift further "
        f"horizontally between the two views. {eye_note}"
    )

    if layout == "side_by_side":
        return [
            header + " Layout: single frame, LEFT eye on the left half, RIGHT eye "
            "on the right half, separated by a dark divider.",
            Image(data=_side_by_side(lpath, rpath, max_width), format="jpeg"),
        ]

    return [
        header + " Layout: two separate images, LEFT eye first, RIGHT eye second.",
        Image(data=_encode(lpath, max_width), format="jpeg"),
        Image(data=_encode(rpath, max_width), format="jpeg"),
    ]


def _capture_pair_bgr(settle_ms: int):
    """Capture both sensors in a FIXED order: (camera 1, camera 0).

    Which of these is the left eye is a calibration result, not an assumption -
    see vision.measure_offset.
    """
    p1, p0 = _scratch("an_cam1.jpg"), _scratch("an_cam0.jpg")
    _capture_both(settle_ms, p1, p0)
    img1, img0 = cv2.imread(p1), cv2.imread(p0)
    if img1 is None or img0 is None:
        raise RuntimeError("captured frames could not be decoded")
    return img1, img0


def _require_calibration() -> dict:
    """Load calibration, refusing to guess if the rig has never been calibrated."""
    cal = vision.load_calibration()
    if "y_shift_px" not in cal or "left_camera" not in cal:
        raise RuntimeError(
            "The stereo rig is not calibrated yet, so distances cannot be "
            "computed. Point the camera at an ordinary cluttered scene - "
            "objects at a range of distances, not a blank or striped wall - "
            "and call calibrate_stereo first."
        )
    if not cal.get("trustworthy", False):
        raise RuntimeError(
            "The stored calibration was measured against a scene that could not "
            "support stereo matching, so any distance from it would be fiction. "
            "Point the camera at ordinary clutter and call calibrate_stereo again."
        )
    return cal


def _left_right(settle_ms: int):
    """Capture and return the pair already ordered (left eye, right eye)."""
    cal = _require_calibration()
    img1, img0 = _capture_pair_bgr(settle_ms)
    if cal["left_camera"] == 1:
        return img1, img0, float(cal["y_shift_px"]), float(cal.get("x_offset_px", 0.0))
    return img0, img1, float(cal["y_shift_px"]), float(cal.get("x_offset_px", 0.0))


@mcp.tool()
def describe_scene(conf_threshold: float = 0.35, settle_ms: int = 1200) -> str:
    """Look through the stereo camera and describe what is there, in text.

    Detects objects and measures how far away each one is, so a model that
    cannot see images can still reason about the scene. This is the main tool -
    prefer it for questions like "what do you see", "what is in front of you",
    "how far away is the nearest thing", or "is the way ahead clear".

    Recognises 80 common categories (people, furniture, vehicles, animals,
    kitchen and household items). Anything outside that list will not be named,
    though it still shows up in the distance measurements.

    Args:
        conf_threshold: Minimum detection confidence, 0-1. Lower finds more
            objects but invents more. 0.35 is a sensible default.
        settle_ms: Auto-exposure settle time in milliseconds before capture.
    """
    try:
        left, right, y_shift, x_off = _left_right(settle_ms)
    except RuntimeError as exc:
        # Depth is unavailable, but detection never needed calibration. Answer
        # the part that still works rather than refusing the whole question,
        # and say plainly which part is missing.
        img1, _ = _capture_pair_bgr(settle_ms)
        return (vision.describe_without_depth(img1, conf_threshold=conf_threshold)
                + f"\n\nWhy depth is missing: {exc}")
    return vision.describe(left, right, y_shift, x_off, conf_threshold=conf_threshold)


@mcp.tool()
def depth_grid(rows: int = 3, cols: int = 3, settle_ms: int = 1200) -> str:
    """Measure distance across a grid of the camera's view, without naming objects.

    Use this when the question is about space and clearance rather than about
    what things are: how far the walls are, which direction is most open,
    whether something is close. Works on any surface with visible texture,
    including objects the detector does not know.

    Args:
        rows: Grid rows, top to bottom.
        cols: Grid columns, left to right.
        settle_ms: Auto-exposure settle time in milliseconds before capture.
    """
    try:
        left, right, y_shift, x_off = _left_right(settle_ms)
    except RuntimeError as exc:
        return f"Cannot measure distances right now.\n\n{exc}"
    return vision.depth_grid(left, right, y_shift, x_off, rows=rows, cols=cols)


@mcp.tool()
def calibrate_stereo(left_camera: int = -1, settle_ms: int = 1500) -> str:
    """Re-measure the alignment between the two sensors and save it.

    The cameras are not mounted perfectly level, and the depth maths corrects
    for that with a stored vertical offset. Run this if distances look wrong,
    or after anything has been physically moved or re-seated. Point the camera
    at a cluttered, well-lit scene - a blank wall has too little texture.

    Args:
        left_camera: Which sensor is physically the LEFT eye, 0 or 1. Leave at
            -1 to infer it from the images. Inference is unreliable on this rig,
            so if you know the physical layout, say so - a declared value is
            always preferred over a guess. To find out: cover one lens and see
            which feed goes dark on the video stream.
        settle_ms: Auto-exposure settle time in milliseconds before capture.
    """
    img1, img0 = _capture_pair_bgr(settle_ms)
    cal = vision.measure_offset(img1, img0,
                                force_left=left_camera if left_camera in (0, 1) else None)
    vision.save_calibration(cal)

    sc = cal["scene"]
    verdict = ("USABLE - distances can be computed." if cal["trustworthy"]
               else "NOT USABLE - this scene cannot support stereo matching, so "
                    "the numbers below are not trustworthy and depth tools will "
                    "refuse to run. Re-aim and repeat.")
    lines = [
        "Calibration result: " + verdict,
        "",
        "Left eye is camera {}, right eye is camera {} (determined from the images, not assumed).".format(
            cal["left_camera"], cal["right_camera"]),
        "Vertical offset: {:+.1f} px, residual row error {:.1f} px".format(
            cal["y_shift_px"], cal["residual_px"]),
        "Row-consistent matches: {} ({:.0f}% of candidates rejected as ambiguous)".format(
            cal["inliers"], cal["ambiguity_rejected_frac"] * 100),
        "Horizontal offset at infinity: {:+.1f} px (removed before measuring depth)".format(
            cal["x_offset_px"]),
        "Depth variation across scene: {:.1f} px of disparity spread {}".format(
            cal["disparity_spread_px"],
            "- OK" if cal["has_depth_variation"] else "- TOO FLAT, everything in view is effectively the same distance"),
        "Eye order: {} ({})".format(
            "settled" if cal["eye_order_decisive"] else "INCONCLUSIVE - pass left_camera=0 or 1 to declare it",
            cal["eye_order_source"]),
        "",
        "Scene check:",
        "  horizontal texture: {:.1f}% of pixels (need >4%, only vertical edges constrain depth)".format(
            sc["horizontal_texture_frac"] * 100),
        "  distinctiveness: {:.2f} (need >0.35, lower means repeating patterns that mismatch)".format(
            sc["distinctiveness"]),
        "",
        "Saved " + cal["measured_at"],
    ]
    return "\n".join(lines)



# --- Ultrasonic distance sensor (HC-SR04) -----------------------------------
# Merged in from the former mcp-pi server so a single MCP endpoint exposes both
# the stereo camera and the ultrasonic sensor, letting the model combine visual
# and acoustic distance in one step. The sensor drives the RP1 GPIO through
# lgpio (RPi.GPIO does not work on the Pi 5). Hardware access is serialised by a
# threading.Lock inside the Ultraschall class - required because mcp 2.x runs
# synchronous tool handlers in a worker-thread pool.
import statistics

from ultraschall import Ultraschall

try:
    _us = Ultraschall()
    _us_error = None
except Exception as exc:  # noqa: BLE001 - report at call time; don't sink the camera tools
    _us = None
    _us_error = str(exc)


@mcp.tool()
def ultraschall_abstand() -> dict:
    """Misst den Abstand zum naechsten Objekt vor dem Ultraschallsensor in Zentimetern.

    Reichweite 2 bis 400 cm, Messkegel ca. 15 Grad - der Sensor meldet das naechste
    Objekt irgendwo in diesem Kegel, nicht zwingend geradeaus. Anders als die
    Kamera-Tiefentools braucht er keine Kalibrierung und funktioniert auch im
    Dunkeln, liefert aber nur EINEN Abstand (den naechsten Reflektor), keine
    Richtung und keine Objektnamen. Fuer 'was ist da' oder eine Tiefenkarte ueber
    das Bild describe_scene bzw. depth_grid nehmen.

    Bei 'zuverlaessig: false' streuen die Messungen stark; dann sind meist mehrere
    Objekte im Kegel oder die Flaeche ist schraeg bzw. schallschluckend.
    """
    if _us is None:
        return {"fehler": f"Ultraschallsensor nicht initialisiert: {_us_error}"}
    werte = _us.messen(5)
    if len(werte) < 3:
        return {"fehler": f"nur {len(werte)} von 5 Messungen gueltig - Sensor pruefen"}
    streuung = max(werte) - min(werte)
    return {
        "abstand_cm": round(statistics.median(werte), 1),
        "zuverlaessig": streuung < 5.0,
        "streuung_cm": round(streuung, 1),
        "messungen": len(werte),
    }


@mcp.tool()
def weg_frei(mindestabstand_cm: float = 50.0) -> str:
    """Prueft mit dem Ultraschallsensor, ob geradeaus bis zum angegebenen Abstand nichts steht.

    Args:
        mindestabstand_cm: Abstand in Zentimetern, bis zu dem der Weg frei sein soll.
    """
    d = ultraschall_abstand()
    if "fehler" in d:
        return f"unbekannt: {d['fehler']}"
    if d["abstand_cm"] >= mindestabstand_cm:
        return f"frei - naechstes Objekt bei {d['abstand_cm']} cm"
    return f"blockiert - Objekt bei {d['abstand_cm']} cm"


if __name__ == "__main__":
    # Host/port live on run() in mcp 2.x, not on the constructor.
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        transport_security=TransportSecuritySettings(
            allowed_hosts=ALLOWED_HOSTS + [f"{h}:{PORT}" for h in ALLOWED_HOSTS],
            allowed_origins=["*"],
        ),
    )
