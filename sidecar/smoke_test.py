"""Smoke test — validates the native pipeline port end-to-end on CPU (no webcam,
no browser). Not a unit test of accuracy; checks the math runs + shapes line up.
Run: cd sidecar && uv run python smoke_test.py
"""
import struct
import numpy as np
import pose_sidecar as P


def main():
    providers = P.select_providers("cpu")
    print(f"[smoke] providers: {providers}")
    # variant 'x' = the only file on disk locally; CPU intentional for the smoke test.
    pipe = P.PosePipeline(providers, variant="x", yolo_res=320, allow_cpu=True)
    print(f"[smoke] auto-read rtmw res {pipe.res_w}x{pipe.res_h}, outputs {pipe.out_x},{pipe.out_y},{pipe.out_z}")

    # synthetic 480x640 RGB frame (no real person — detect may return []).
    h, w = 480, 640
    frame = np.random.randint(0, 255, (h, w, 3), np.uint8)

    boxes = pipe.detect(frame, 0.3)
    print(f"[smoke] detect ran → {len(boxes)} person box(es)")

    # force a centered box so the pose+decode path is exercised regardless of detect.
    forced = [(w * 0.25, h * 0.1, w * 0.5, h * 0.8, 0.99)]
    frame_out, timings = pipe.infer(frame, 0.3, forced)
    assert frame_out is not None, "infer returned None for a forced box"
    k3 = frame_out["keypoints3D"]
    k2 = frame_out["keypoints2D"]
    assert len(k3) == 133, f"expected 133 keypoints3D, got {len(k3)}"
    assert len(k2) == 133, f"expected 133 keypoints2D, got {len(k2)}"
    for p in k3[:1] + k2[:1]:
        for key, v in p.items():
            assert np.isfinite(v), f"non-finite {key}={v}"
    print(f"[smoke] pose+decode ran → 133 kpts, stage ms: "
          f"det={timings['detect']:.1f} pre={timings['preprocess']:.1f} "
          f"inf={timings['inference']:.1f} dec={timings['decode']:.1f}")

    # header pack/unpack matches the JS provider (ts f64 | w u32 | h u32 | everyN u32).
    packed = struct.pack("<dIII", 123.0, w, h, 2)
    ts, ww, hh, n = struct.unpack_from("<dIII", packed, 0)
    assert (ww, hh, n) == (w, h, 2) and ts == 123.0, "header round-trip mismatch"
    assert len(packed) == 20, "header must be 20 bytes (matches JS HEADER_BYTES)"
    print("[smoke] wire header round-trips ✓")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
