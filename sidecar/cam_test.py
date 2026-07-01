"""Standalone camera viewer — SEE what each camera actually delivers. No models, no
sidecar, no GPU. Pure OpenCV frame grab.

opencv-python-headless has NO GUI (no cv2.imshow window), so this SAVES frames to JPG
next to where you run it — open the files to look. Burst mode grabs several frames so
you can tell a live stream (frames change) from a frozen/black one.

  uv run --project sidecar python sidecar/cam_test.py          # 1 snapshot of devices 0..7
  uv run --project sidecar python sidecar/cam_test.py 5        # device 5, burst of 5 frames
  uv run --project sidecar python sidecar/cam_test.py 5 20     # device 5, 20 frames (see motion)

Or via npm:  npm run cam:test -- 5
"""
import os
import sys
import time

import cv2


def _backends():
    return ([(cv2.CAP_DSHOW, "DShow"), (cv2.CAP_MSMF, "MSMF")]
            if sys.platform == "win32" else [(cv2.CAP_ANY, "default")])


def grab(idx, n=1):
    """Open device idx, warm up, save n frames as cam-<idx>[-k].jpg. Reports res + mean
    brightness (0=black … 255=white) + the file path so you can open and look."""
    for be, name in _backends():
        cap = cv2.VideoCapture(idx, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # latest-frame-wins
        ok, f = False, None
        for _ in range(10):  # warm up — first frames can be empty
            ok, f = cap.read()
            if ok and f is not None:
                break
            time.sleep(0.05)
        if not ok or f is None:
            cap.release()
            print(f"  device {idx} ({name}): opened but no frames")
            continue
        saved = []
        for k in range(n):
            ok, f = cap.read()
            if not ok or f is None:
                break
            suffix = "" if n == 1 else f"-{k:02d}"
            path = os.path.abspath(f"cam-{idx}{suffix}.jpg")
            cv2.imwrite(path, f)
            saved.append((path, f.shape[1], f.shape[0], float(f.mean())))
            time.sleep(0.05)
        cap.release()
        for path, w, h, mean in saved:
            tag = "BLACK/dark" if mean <= 5 else ("dark" if mean < 20 else "OK")
            print(f"  device {idx} ({name}): {w}x{h} mean={mean:5.1f} [{tag}] -> {path}")
        return True
    print(f"  device {idx}: no frames on any backend")
    return False


def main():
    args = sys.argv[1:]
    if args:
        idx = int(args[0])
        n = int(args[1]) if len(args) > 1 else 5
        print(f"[cam_test] device {idx}, {n} frame(s) — open the cam-{idx}*.jpg files to LOOK")
        grab(idx, n)
    else:
        print("[cam_test] snapshotting devices 0..7 — open the cam-N.jpg files to LOOK")
        for i in range(8):
            grab(i, 1)
    print("[cam_test] done. Open the cam-*.jpg files to SEE each feed.")


if __name__ == "__main__":
    main()
