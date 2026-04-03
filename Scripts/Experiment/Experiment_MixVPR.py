"""
Standalone Loop-Closure Detection Test — KITTI Sequence 07
===========================================================

Tests the MixVPR (and NetVLAD fallback) loop-closure detectors on real
KITTI images.  Sequence 07 revisits the starting location around frame
1060, making it a clean ground-truth loop for validation.

Usage
-----
    python test_loop_closure.py                         # defaults
    python test_loop_closure.py --seq-dir /path/to/07/image_2
    python test_loop_closure.py --detector netvlad      # fallback test
    python test_loop_closure.py --stride 5 --top-k 5    # denser sampling

Requirements
------------
    torch, torchvision, opencv-python, numpy, matplotlib (optional)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Attempt to import project detectors; fall back to defining them inline
#  so the test runs fully standalone.
# ---------------------------------------------------------------------------
try:
    from Module.LoopClosure.NetVLAD_MixVPR import (
        MixVPRLoopDetector,
        NetVLADLoopDetectora,  # Note: renamed to avoid conflict with original NetVLADLoopDetector
        GeometricVerifier,
    )
    _HAS_PROJECT_IMPORTS = True
except ImportError:
    _HAS_PROJECT_IMPORTS = False
    print("[warn] Could not import from Module.LoopClosure — "
          "will attempt standalone class definitions.\n"
          "       Run from the project root or set PYTHONPATH.")
    sys.exit(1)


# ===================================================================
#  Helpers
# ===================================================================

def load_image_as_tensor(path: str) -> torch.Tensor:
    """Load a single image → (1, 3, H, W) float32 tensor in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t


def collect_image_paths(
    seq_dir: str, stride: int = 1
) -> list[tuple[int, str]]:
    """Return sorted list of (frame_index, path) with given stride."""
    patterns = [os.path.join(seq_dir, f"*.{ext}") for ext in ("png", "jpg")]
    paths = sorted(sum((glob.glob(p) for p in patterns), []))
    if not paths:
        raise FileNotFoundError(f"No images found in {seq_dir}")
    result = []
    for i, p in enumerate(paths):
        if i % stride == 0:
            result.append((i, p))
    return result


def similarity_matrix_figure(
    sim_matrix: np.ndarray,
    frame_indices: list[int],
    title: str,
    save_path: str,
) -> None:
    """Save a similarity-matrix heatmap (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed — cannot save heatmap")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="hot", interpolation="nearest",
                   vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    # Tick labels: show every N-th frame index
    n = len(frame_indices)
    step = max(1, n // 15)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([frame_indices[t] for t in ticks], rotation=45, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([frame_indices[t] for t in ticks], fontsize=7)
    ax.set_xlabel("Keyframe index")
    ax.set_ylabel("Keyframe index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → heatmap saved to {save_path}")


# ===================================================================
#  Core test routines
# ===================================================================

def test_pairwise_similarity(
    detector: MixVPRLoopDetector | NetVLADLoopDetectora,
    img_a_path: str,
    img_b_path: str,
    label: str = "",
) -> float:
    """Encode two images and print their cosine similarity."""
    a = load_image_as_tensor(img_a_path)
    b = load_image_as_tensor(img_b_path)
    da = detector._encode(a)
    db = detector._encode(b)
    sim = float((da @ db.T).squeeze())
    tag = f"  [{label}]" if label else ""
    print(f"  sim({Path(img_a_path).stem}, {Path(img_b_path).stem})"
          f" = {sim:.4f}{tag}")
    return sim


def test_full_sequence(
    detector_name: str,
    seq_dir: str,
    stride: int,
    min_gap: int,
    top_k: int,
    threshold: float,
    expected_loop: tuple[int, int],
    save_heatmap: bool,
    device: str,
) -> bool:
    """
    Full integration test:
      1. Build the detector.
      2. Feed every stride-th frame as a keyframe.
      3. Query for loop closures at each step.
      4. Check whether the expected loop pair is detected.
      5. Optionally save a descriptor-similarity heatmap.
    """
    print(f"\n{'='*70}")
    print(f"  Detector : {detector_name}")
    print(f"  Sequence : {seq_dir}")
    print(f"  Stride   : {stride}   |  min_gap : {min_gap}  |  "
          f"top_k : {top_k}  |  threshold : {threshold:.2f}")
    print(f"  Expected loop : frame {expected_loop[0]} ↔ {expected_loop[1]}")
    print(f"{'='*70}\n")

    # ── Build detector ──────────────────────────────────────────────
    if detector_name == "mixvpr":
        cfg = SimpleNamespace(
            device=device, threshold=threshold, top_k=top_k,
            min_gap=min_gap, weight_path=None,
            orb_features=1000, match_ratio=0.75,
            ransac_threshold=1.0,
            geometric_min_inliers=30,
            geometric_min_inlier_ratio=0.25,
        )
        detector = MixVPRLoopDetector(cfg)
    else:
        cfg = SimpleNamespace(
            device=device, backbone="resnet18",
            descriptor_dim=512, threshold=threshold,
            top_k=top_k, min_gap=min_gap,
        )
        detector = NetVLADLoopDetectora(cfg)

    # ── Collect images ──────────────────────────────────────────────
    images = collect_image_paths(seq_dir, stride)
    n = len(images)
    print(f"  Loaded {n} keyframes (of {n * stride} total frames)\n")

    # ── Pairwise sanity check on the expected pair ──────────────────
    print("  ── Pairwise similarity sanity checks ──")
    loop_a_path = os.path.join(seq_dir, f"{expected_loop[0]:06d}.png")
    loop_b_path = os.path.join(seq_dir, f"{expected_loop[1]:06d}.png")

    if os.path.exists(loop_a_path) and os.path.exists(loop_b_path):
        sim_loop = test_pairwise_similarity(
            detector, loop_a_path, loop_b_path, "expected loop pair")
    else:
        sim_loop = -1.0
        print(f"  [warn] Could not load pair images for pairwise check")

    # Pick a non-loop frame roughly in the middle of the sequence
    mid_idx = len(images) // 2
    _, mid_path = images[mid_idx]
    if os.path.exists(loop_a_path):
        test_pairwise_similarity(
            detector, loop_a_path, mid_path, "non-loop baseline")
    print()

    # ── Feed keyframes + query ──────────────────────────────────────
    all_descriptors: list[torch.Tensor] = []
    frame_indices: list[int] = []
    detected_loops: list[tuple[int, int, float]] = []

    t0 = time.perf_counter()
    for step, (fidx, fpath) in enumerate(images):
        img = load_image_as_tensor(fpath)
        detector.add_keyframe(fidx, img)
        all_descriptors.append(detector._encode(img))
        frame_indices.append(fidx)

        candidates = detector.query(fidx, min_gap=min_gap)
        for cand_idx, cand_sim in candidates:
            detected_loops.append((fidx, cand_idx, cand_sim))

        # Progress
        if (step + 1) % 50 == 0 or step == n - 1:
            elapsed = time.perf_counter() - t0
            print(f"  [{step+1:>4d}/{n}]  frame {fidx:>6d}  |  "
                  f"DB size {len(detector.kf_indices):>4d}  |  "
                  f"loops so far: {len(detected_loops):>3d}  |  "
                  f"{elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n  Encoding + querying done in {elapsed:.1f}s  "
          f"({elapsed/n*1000:.1f} ms/frame)")

    # ── Results ─────────────────────────────────────────────────────
    print(f"\n  ── Detected loop closures ({len(detected_loops)}) ──")
    if detected_loops:
        # Sort by similarity descending
        detected_loops.sort(key=lambda x: -x[2])
        for q, m, s in detected_loops[:20]:
            marker = " ◀ EXPECTED" if (
                _is_expected_match(q, m, expected_loop, tolerance=stride*2)
            ) else ""
            print(f"    KF {q:>6d} ↔ {m:>6d}   sim={s:.4f}{marker}")
        if len(detected_loops) > 20:
            print(f"    ... and {len(detected_loops) - 20} more")
    else:
        print("    (none)")

    # ── Check expected loop ─────────────────────────────────────────
    found_expected = any(
        _is_expected_match(q, m, expected_loop, tolerance=stride * 2)
        for q, m, _ in detected_loops
    )

    print(f"\n  ── Verdict ──")
    if found_expected:
        best = max(
            (s for q, m, s in detected_loops
             if _is_expected_match(q, m, expected_loop, tolerance=stride*2)),
            default=0.0)
        print(f"  ✅  PASS — expected loop {expected_loop[0]}↔{expected_loop[1]} "
              f"detected (best sim={best:.4f})")
    else:
        print(f"  ❌  FAIL — expected loop {expected_loop[0]}↔{expected_loop[1]} "
              f"NOT detected")
        if sim_loop > 0:
            print(f"         pairwise sim was {sim_loop:.4f}, "
                  f"threshold is {threshold:.2f}")

    # ── Heatmap ─────────────────────────────────────────────────────
    if save_heatmap and all_descriptors:
        descs = torch.cat(all_descriptors, dim=0)            # (N, D)
        sim_mat = (descs @ descs.T).numpy()
        out_name = f"sim_heatmap_{detector_name}_seq07.png"
        similarity_matrix_figure(
            sim_mat, frame_indices,
            title=f"{detector_name.upper()} — KITTI Seq 07  "
                  f"(stride={stride}, threshold={threshold})",
            save_path=out_name)

    # ── Geometric verification test on best loop ────────────────────
    if (found_expected
            and detector_name == "mixvpr"
            and hasattr(detector, "geometric_verify")):
        print(f"\n  ── Geometric verification (ORB + E-matrix) ──")
        best_loop = max(
            ((q, m, s) for q, m, s in detected_loops
             if _is_expected_match(q, m, expected_loop, tolerance=stride*2)),
            key=lambda x: x[2])
        q_idx, m_idx, _ = best_loop

        # Use a dummy K (KITTI seq07 approximate intrinsics)
        K = np.array([
            [707.0912, 0.0,      601.8873],
            [0.0,      707.0912, 183.1104],
            [0.0,      0.0,      1.0     ],
        ], dtype=np.float64)

        accepted, inlier_ratio, T_4x4 = detector.geometric_verify(
            q_idx, m_idx, K)
        if accepted:
            print(f"  ✅  Geometric verification PASSED  "
                  f"(inlier_ratio={inlier_ratio:.3f})")
            if T_4x4 is not None:
                t_vec = T_4x4[:3, 3]
                print(f"      translation = [{t_vec[0]:.4f}, "
                      f"{t_vec[1]:.4f}, {t_vec[2]:.4f}]")
        else:
            print(f"  ⚠️  Geometric verification FAILED  "
                  f"(inlier_ratio={inlier_ratio:.3f})")

    return found_expected


def _is_expected_match(
    q: int, m: int, expected: tuple[int, int], tolerance: int
) -> bool:
    """Check if (q, m) matches the expected loop pair within tolerance."""
    a, b = expected
    return (
        (abs(q - a) <= tolerance and abs(m - b) <= tolerance) or
        (abs(q - b) <= tolerance and abs(m - a) <= tolerance)
    )


# ===================================================================
#  Additional unit-style tests
# ===================================================================

def test_detector_reset(detector_name: str, device: str) -> None:
    """Ensure .reset() clears the database completely."""
    print("\n  ── Test: detector reset ──")
    if detector_name == "mixvpr":
        cfg = SimpleNamespace(
            device=device, threshold=0.5, top_k=3, min_gap=5,
            weight_path=None)
        det = MixVPRLoopDetector(cfg)
    else:
        cfg = SimpleNamespace(
            device=device, backbone="resnet18",
            descriptor_dim=512, threshold=0.5,
            top_k=3, min_gap=5)
        det = NetVLADLoopDetectora(cfg)

    dummy = torch.rand(1, 3, 224, 224)
    det.add_keyframe(0, dummy)
    det.add_keyframe(10, dummy)
    assert len(det.kf_indices) == 2, "Expected 2 keyframes before reset"

    det.reset()
    assert len(det.kf_indices) == 0, "Expected 0 keyframes after reset"
    assert len(det.descriptors) == 0, "Expected 0 descriptors after reset"
    print("  ✅  reset() works correctly")


def test_min_gap_filtering(detector_name: str, device: str) -> None:
    """Verify that min_gap filtering excludes temporally close frames."""
    print("\n  ── Test: min_gap filtering ──")
    if detector_name == "mixvpr":
        cfg = SimpleNamespace(
            device=device, threshold=0.0, top_k=10, min_gap=50,
            weight_path=None)
        det = MixVPRLoopDetector(cfg)
    else:
        cfg = SimpleNamespace(
            device=device, backbone="resnet18",
            descriptor_dim=512, threshold=0.0,
            top_k=10, min_gap=50)
        det = NetVLADLoopDetectora(cfg)

    # Same image at indices 0, 10, 20, 100 — only 100 should match 0
    img = torch.rand(1, 3, 224, 224)
    for idx in [0, 10, 20, 100]:
        det.add_keyframe(idx, img)

    results = det.query(0, min_gap=50)
    returned_indices = {r[0] for r in results}

    assert 10 not in returned_indices, "idx=10 should be filtered (gap=10 < 50)"
    assert 20 not in returned_indices, "idx=20 should be filtered (gap=20 < 50)"
    assert 100 in returned_indices, "idx=100 should NOT be filtered (gap=100 >= 50)"
    print("  ✅  min_gap filtering works correctly")


def test_self_similarity(detector_name: str, device: str) -> None:
    """Same image should have similarity ≈ 1.0."""
    print("\n  ── Test: self-similarity ──")
    if detector_name == "mixvpr":
        cfg = SimpleNamespace(
            device=device, threshold=0.5, top_k=3, min_gap=1,
            weight_path=None)
        det = MixVPRLoopDetector(cfg)
    else:
        cfg = SimpleNamespace(
            device=device, backbone="resnet18",
            descriptor_dim=512, threshold=0.5,
            top_k=3, min_gap=1)
        det = NetVLADLoopDetectora(cfg)

    img = torch.rand(1, 3, 320, 320)
    d = det._encode(img)
    self_sim = float((d @ d.T).squeeze())
    print(f"  self-similarity = {self_sim:.6f}")
    assert abs(self_sim - 1.0) < 1e-4, f"Expected ~1.0, got {self_sim}"
    print("  ✅  self-similarity ≈ 1.0")


def test_different_images(detector_name: str, device: str) -> None:
    """Random images should have low similarity."""
    print("\n  ── Test: different-image similarity ──")
    if detector_name == "mixvpr":
        cfg = SimpleNamespace(
            device=device, threshold=0.5, top_k=3, min_gap=1,
            weight_path=None)
        det = MixVPRLoopDetector(cfg)
    else:
        cfg = SimpleNamespace(
            device=device, backbone="resnet18",
            descriptor_dim=512, threshold=0.5,
            top_k=3, min_gap=1)
        det = NetVLADLoopDetectora(cfg)

    torch.manual_seed(42)
    a = det._encode(torch.rand(1, 3, 320, 320))
    b = det._encode(torch.rand(1, 3, 320, 320))
    sim = float((a @ b.T).squeeze())
    print(f"  random-pair similarity = {sim:.4f}")
    assert sim < 0.9, f"Random images too similar: {sim}"
    print("  ✅  random images have low similarity")


# ===================================================================
#  CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test loop-closure detection on KITTI sequence 07")
    parser.add_argument(
        "--seq-dir", type=str,
        default="data/KITTI_odometry/dataset/sequences/07/image_2",
        help="Path to KITTI image_2 folder")
    parser.add_argument(
        "--detector", choices=["mixvpr", "netvlad"], default="mixvpr",
        help="Which detector to test (default: mixvpr)")
    parser.add_argument(
        "--stride", type=int, default=5,
        help="Use every N-th frame as a keyframe (default: 5)")
    parser.add_argument(
        "--min-gap", type=int, default=50,
        help="Minimum frame-index gap for LC candidates (default: 50)")
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top candidates to return per query (default: 5)")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="VPR similarity threshold (default: 0.70 mixvpr / 0.80 netvlad)")
    parser.add_argument(
        "--loop-a", type=int, default=0,
        help="Frame index of expected loop end A (default: 0)")
    parser.add_argument(
        "--loop-b", type=int, default=1060,
        help="Frame index of expected loop end B (default: 1060)")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Save a descriptor-similarity heatmap PNG")
    parser.add_argument(
        "--skip-unit", action="store_true",
        help="Skip the fast unit tests, run only the sequence test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold
    if threshold is None:
        threshold = 0.70 if args.detector == "mixvpr" else 0.80

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ── Fast unit tests (no images needed) ──────────────────────────
    if not args.skip_unit:
        print(f"\n{'='*70}")
        print(f"  Unit tests — {args.detector}")
        print(f"{'='*70}")
        test_self_similarity(args.detector, device)
        test_different_images(args.detector, device)
        test_detector_reset(args.detector, device)
        test_min_gap_filtering(args.detector, device)
        print(f"\n  All unit tests passed ✅")

    # ── Full sequence test ──────────────────────────────────────────
    if not os.path.isdir(args.seq_dir):
        print(f"\n[skip] Sequence directory not found: {args.seq_dir}")
        print(f"       Provide --seq-dir /path/to/07/image_2 to run "
              f"the full sequence test.")
        return

    passed = test_full_sequence(
        detector_name=args.detector,
        seq_dir=args.seq_dir,
        stride=args.stride,
        min_gap=args.min_gap,
        top_k=args.top_k,
        threshold=threshold,
        expected_loop=(args.loop_a, args.loop_b),
        save_heatmap=args.heatmap,
        device=device,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()