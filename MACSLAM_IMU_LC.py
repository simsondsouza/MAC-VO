"""
MAC-SLAM + IMU  entry point
============================
Combines MAC-VO + IMU PVGO (from Odometry/MACVO_IMU.py) with
Loop Closure + Global PGO (from Odometry/MACSLAM.py) into a single system.

Architecture
------------
* **Local VO**  — MACVO_IMU: FlowFormerCov visual tracking + sliding-window PVGO
* **Loop Closure** — MixVPR place recognition + ORB geometric verification
* **Global PGO**   — asynchronous switchable-constraint pose-graph optimiser

Usage
-----
::

    python MACSLAM_IMU_LC.py \\
        --odom Config/Experiment/MACSLAM/MACSLAM_IMU_LC.yaml \\
        --data Config/Sequence/KITTI_K07.yaml \\
        [--noeval] [--timing] [--preload]

The script auto-upgrades ``type: KITTI`` to ``type: KITTI_IMU`` so plain
KITTI sequence configs can be passed directly.
"""

import argparse
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace

from DataLoader import SequenceBase, smart_transform
from DataLoader.Interface import StereoInertialFrame
from DataLoader.Dataset.KITTI_IMU import KITTI_IMU_Sequence  # registers "KITTI_IMU"

from Odometry.MACVO_IMU import MACVO_IMU
from Odometry.MACSLAM import MACSLAM, _LOOP_DETECTOR_REGISTRY
from Module.Optimization.GlobalPGO.Optimizer import GlobalPGO_Optimizer
from Evaluation.EvalSeq import EvaluateSequences

from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Visualize import fig_plt, rr_plt
from Utility.Timer import Timer

try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════

def StatusCallback(
    frame: StereoInertialFrame, system: MACSLAM, pb: ColoredTqdm,
) -> None:
    vo = system.local_vo
    vram = "N/A"
    if torch.cuda.is_available():
        vram = f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
    pgo = system.global_pgo
    pb.set_description(
        f"{vo.graph} | "
        f"seq={pgo.num_sequential} lc={pgo.num_loop_closures} "
        f"GPG={'RUN' if pgo.is_running else 'IDLE'} | "
        f"VRAM={vram}"
    )


def VisualizeRerunCallback(
    frame: StereoInertialFrame, system: MACSLAM, pb: ColoredTqdm,
) -> None:
    if not HAS_RERUN:
        return
    rr.set_time_sequence("frame_idx", frame.frame_idx)
    vo = system.local_vo
    if vo.graph.frames.data["need_interp"][-1]:
        return
    if frame.frame_idx > 0:
        rr_plt.log_trajectory("/world/est",
                              pp.SE3(vo.graph.frames.data["pose"].tensor))
    rr_plt.log_camera("/world/macvo/cam_left",
                      pp.SE3(vo.graph.frames.data["pose"][-1]),
                      vo.graph.frames.data["K"][-1])
    rr_plt.log_image("/world/macvo/cam_left",
                     frame.stereo.imageL[0].permute(1, 2, 0))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAC-SLAM + IMU: loop closure + IMU PVGO on KITTI"
    )
    p.add_argument("--odom", type=str, required=True,
                   help="Odometry config YAML (e.g. Config/Experiment/MACSLAM/MACSLAM_IMU_LC.yaml). "
                        "If it contains a 'Data:' section, --data can be omitted.")
    p.add_argument("--data", type=str, default=None,
                   help="Sequence config YAML. Plain KITTI configs are auto-upgraded to KITTI_IMU.")
    p.add_argument("--seq_from",    type=int, default=0)
    p.add_argument("--seq_to",      type=int, default=None)
    p.add_argument("--resultRoot",  type=str, default="./Results")
    p.add_argument("--useRR",       action="store_true")
    p.add_argument("--saveplt",     action="store_true")
    p.add_argument("--preload",     action="store_true")
    p.add_argument("--autoremove",  action="store_true")
    p.add_argument("--noeval",      action="store_true")
    p.add_argument("--timing",      action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# System factory  (replicates MACSLAM.from_config but uses MACVO_IMU)
# ═══════════════════════════════════════════════════════════════════════

def build_system(cfg: SimpleNamespace) -> MACSLAM:
    """Build a MACSLAM instance whose local_vo is MACVO_IMU."""

    # ── Local VO with IMU PVGO ───────────────────────────────────────────
    local_vo = MACVO_IMU.from_config(cfg)

    # ── SLAM-specific settings (with safe defaults) ──────────────────────
    slam_cfg = getattr(cfg, "SLAM", None)
    if slam_cfg is None:
        slam_cfg = SimpleNamespace(
            loop_closure=SimpleNamespace(
                type="MixVPRLoopDetector",
                args=SimpleNamespace(
                    device=("cuda" if torch.cuda.is_available() else "cpu"),
                    threshold=0.5,
                    top_k=3,
                    min_gap=80,
                    orb_features=1000,
                    match_ratio=0.75,
                    ransac_threshold=1.0,
                    geometric_min_inliers=30,
                    geometric_min_inlier_ratio=0.25,
                ),
            ),
            global_pgo=SimpleNamespace(
                lambda_switch=1.0,
                max_iterations=50,
                huber_delta=0.5,
                device="cpu",
            ),
            lc_min_gap=80,
            lc_geometric_threshold=0.25,
            keyframe_freq=5,
            apply_correction=True,
        )

    # ── Loop-closure detector ────────────────────────────────────────────
    lc_cfg      = slam_cfg.loop_closure
    lc_type     = getattr(lc_cfg, "type", "MixVPRLoopDetector")
    lc_cls      = _LOOP_DETECTOR_REGISTRY.get(lc_type)
    if lc_cls is None:
        raise ValueError(
            f"Unknown loop-closure detector '{lc_type}'. "
            f"Available: {list(_LOOP_DETECTOR_REGISTRY)}"
        )
    loop_detector = lc_cls(lc_cfg.args)
    Logger.write("info", f"MACSLAM-IMU-LC: loop detector = {lc_type}")

    # ── Global PGO ──────────────────────────────────────────────────────
    g = slam_cfg.global_pgo
    global_pgo = GlobalPGO_Optimizer(
        lambda_switch  = g.lambda_switch,
        max_iterations = g.max_iterations,
        huber_delta    = getattr(g, "huber_delta", 0.5),
        device         = g.device,
    )

    return MACSLAM(
        local_vo               = local_vo,
        loop_detector          = loop_detector,
        global_pgo             = global_pgo,
        lc_min_gap             = getattr(slam_cfg, "lc_min_gap", 80),
        lc_geometric_threshold = getattr(slam_cfg, "lc_geometric_threshold", 0.25),
        keyframe_freq          = getattr(slam_cfg, "keyframe_freq", 5),
        apply_correction       = getattr(slam_cfg, "apply_correction", True),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = get_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg, cfg_dict         = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]

    # Data config: explicit --data overrides; fall back to Data: section in odom config.
    if args.data is not None:
        datacfg, datacfg_dict = load_config(Path(args.data))
    elif hasattr(cfg, "Data"):
        datacfg, datacfg_dict = cfg.Data, cfg_dict.get("Data", {})
    else:
        raise ValueError(
            "--data is required when the odom config has no 'Data:' section."
        )

    project_name = odomcfg.name + "@" + datacfg.name
    Logger.write("info", f"Project: {project_name}")

    # ── Sandbox ─────────────────────────────────────────────────────────
    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    if args.autoremove:
        exp_space.set_autoremove()
    exp_space.config = {
        "Project":  project_name,
        "Odometry": odomcfg_dict,
        "SLAM":     cfg_dict.get("SLAM", {}),
        "Data":     {"args": datacfg_dict,
                     "start_idx": args.seq_from,
                     "end_idx":   args.seq_to},
    }

    # ── Visualisation ───────────────────────────────────────────────────
    if args.useRR and HAS_RERUN:
        rr_plt.default_mode = "rerun"
        rr_plt.init_connect(project_name)

    Timer.setup(active=args.timing)
    fig_plt.default_mode = "image" if args.saveplt else "none"

    # ── Data source ─────────────────────────────────────────────────────
    # Auto-upgrade plain KITTI → KITTI_IMU so KITTI_K07.yaml works directly.
    seq_type = datacfg.type
    if seq_type == "KITTI":
        seq_type = "KITTI_IMU"
        Logger.write("info", "Sequence type 'KITTI' upgraded to 'KITTI_IMU' for IMU support.")

    sequence = smart_transform(
        SequenceBase[StereoInertialFrame]
            .instantiate(seq_type, datacfg.args)
            .clip(args.seq_from, args.seq_to),
        cfg.Preprocess,
    )

    if args.preload:
        Logger.write("warn", "Preloading a full KITTI sequence may use >20 GB RAM.")
        sequence = sequence.preload()

    # ── Build & run ─────────────────────────────────────────────────────
    system = build_system(asNamespace(exp_space.config))

    def on_frame_finished(
        frame: StereoInertialFrame, sys: MACSLAM, pb: ColoredTqdm,
    ) -> None:
        VisualizeRerunCallback(frame, sys, pb)
        StatusCallback(frame, sys, pb)

    system.receive_frames(sequence, exp_space, on_frame_finished=on_frame_finished)

    # ── Final logging ───────────────────────────────────────────────────
    if HAS_RERUN and args.useRR:
        rr_plt.log_trajectory(
            "/world/est",
            torch.tensor(np.load(exp_space.path("poses.npy"))[:, 1:]))

    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    # ── Evaluate ────────────────────────────────────────────────────────
    if not args.noeval:
        header, result = EvaluateSequences(
            [str(exp_space.folder)], correct_scale=False)
        print_as_table(header, result)

    Logger.write("info", "MAC-SLAM + IMU finished.")
