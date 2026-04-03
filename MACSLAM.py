"""
MAC-SLAM  entry point
=====================

Usage
-----
::

    python MACSLAM.py \\
        --odom Config/Experiment/MACSLAM/MACSLAM_KITTI.yaml \\
        --data Config/Sequence/KITTI_K00.yaml \\
        [--useRR] [--timing] [--preload]

The script mirrors ``MACVO.py`` but builds a ``MACSLAM`` system that
adds loop closure detection and global pose-graph optimisation.
"""

import argparse
import torch
import numpy as np
import pypose as pp
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACSLAM import MACSLAM

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

def VisualizeRerunCallback(
    frame: StereoFrame, system: MACSLAM, pb: ColoredTqdm,
) -> None:
    if not HAS_RERUN:
        return
    rr.set_time_sequence("frame_idx", frame.frame_idx)
    vo = system.local_vo

    if vo.graph.frames.data["need_interp"][-1]:
        return

    if frame.frame_idx > 0:
        rr_plt.log_trajectory(
            "/world/est", pp.SE3(vo.graph.frames.data["pose"].tensor))

    rr_plt.log_camera(
        "/world/macvo/cam_left",
        pp.SE3(vo.graph.frames.data["pose"][-1]),
        vo.graph.frames.data["K"][-1],
    )
    rr_plt.log_image(
        "/world/macvo/cam_left",
        frame.stereo.imageL[0].permute(1, 2, 0),
    )


def StatusCallback(
    frame: StereoFrame, system: MACSLAM, pb: ColoredTqdm,
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAC-SLAM: MAC-VO + Loop Closure")
    p.add_argument("--odom", type=str,
                   default="Config/Experiment/MACSLAM/MACSLAM_KITTI.yaml")
    p.add_argument("--data", type=str,
                   default="Config/Sequence/KITTI_K00.yaml")
    p.add_argument("--seq_to",   type=int, default=None)
    p.add_argument("--seq_from", type=int, default=0)
    p.add_argument("--resultRoot", type=str, default="./Results")
    p.add_argument("--useRR",      action="store_true")
    p.add_argument("--saveplt",    action="store_true")
    p.add_argument("--preload",    action="store_true")
    p.add_argument("--autoremove", action="store_true")
    p.add_argument("--noeval",     action="store_true")
    p.add_argument("--timing",     action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = get_args()

    # ── Config ──────────────────────────────────────────────────────────
    cfg, cfg_dict         = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]
    datacfg, datacfg_dict = load_config(Path(args.data))
    project_name          = odomcfg.name + "@" + datacfg.name

    # ── Sandbox ─────────────────────────────────────────────────────────
    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    if args.autoremove:
        exp_space.set_autoremove()
    exp_space.config = {
        "Project":  project_name,
        "Odometry": odomcfg_dict,
        "SLAM":     cfg_dict.get("SLAM", {}),
        "Data":     {"args": datacfg_dict,
                     "end_idx": args.seq_to,
                     "start_idx": args.seq_from},
    }

    # ── Visualisation ───────────────────────────────────────────────────
    if args.useRR and HAS_RERUN:
        rr_plt.default_mode = "rerun"
        rr_plt.init_connect(project_name)

    Timer.setup(active=args.timing)
    fig_plt.default_mode = "image" if args.saveplt else "none"

    def on_frame_finished(
        frame: StereoFrame, system, pb: ColoredTqdm,
    ) -> None:
        if isinstance(system, MACSLAM):
            VisualizeRerunCallback(frame, system, pb)
            StatusCallback(frame, system, pb)

    # ── Data ────────────────────────────────────────────────────────────
    sequence = smart_transform(
        SequenceBase[StereoFrame]
            .instantiate(datacfg.type, datacfg.args)
            .clip(args.seq_from, args.seq_to),
        cfg.Preprocess,
    )
    if args.preload:
        sequence = sequence.preload()


    # ── Build & run ─────────────────────────────────────────────────────
    system = MACSLAM[StereoFrame].from_config(asNamespace(exp_space.config))
    system.receive_frames(
        sequence, exp_space, on_frame_finished=on_frame_finished)

    # ── Final logging ───────────────────────────────────────────────────
    if HAS_RERUN:
        rr_plt.log_trajectory(
            "/world/est",
            torch.tensor(np.load(exp_space.path("poses.npy"))[:, 1:]))
        try:
            m = system.get_map()
            rr_plt.log_points(
                "/world/point_cloud",
                m.map_points.data["pos_Tw"].tensor,
                m.map_points.data["color"].tensor,
                m.map_points.data["cov_Tw"].tensor,
                "color")
        except RuntimeError:
            Logger.write("warn",
                "Unable to log full pointcloud — is mapping mode on?")

    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    # ── Evaluate ────────────────────────────────────────────────────────
    if not args.noeval:
        header, result = EvaluateSequences(
            [str(exp_space.folder)], correct_scale=False)
        print_as_table(header, result)

    Logger.write("info", "MAC-SLAM finished.")