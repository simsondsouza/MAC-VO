"""
MACVO_IMU.py  –  Entry point for MAC-VO + IMU on KITTI
=======================================================
Mirrors MACVO.py but uses:
  • KITTI_IMU_Sequence  (StereoInertialFrame loader)
  • MACVO_IMU            (visual odometry + sliding-window PVGO)

Usage
-----
python3 MACVO_IMU.py \\
    --odom  Config/Experiment/MACVO/Paper_Reproduce_IMU.yaml \\
    --data  Config/Sequence/KITTI_K00_IMU.yaml

Evaluation is printed automatically after the run and saved in
Results/<project_name>/<timestamp>/.

Comparing against baseline
--------------------------
Run the standard MACVO baseline first:

    python3 MACVO.py \\
        --odom  Config/Experiment/MACVO/Paper_Reproduce.yaml \\
        --data  Config/Sequence/KITTI_K00.yaml

Then evaluate both sandboxes together:

    PYTHONPATH=. python3 Evaluation/EvalSeq.py \\
        --spaces Results/MACVO-PaperReproduce@K00/<ts_baseline> \\
                 Results/MACVO-IMU@K00/<ts_imu>
"""

import argparse
import torch
import numpy as np
import pypose as pp
from pathlib import Path

from DataLoader import SequenceBase, smart_transform
from DataLoader.Interface import StereoInertialFrame
from DataLoader.Dataset.KITTI_IMU import KITTI_IMU_Sequence  # registers "KITTI_IMU"

from Odometry.MACVO_IMU import MACVO_IMU
from Evaluation.EvalSeq import EvaluateSequences

from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Timer import Timer


# ---------------------------------------------------------------------------
# Per-frame callbacks (mirrors MACVO.py)
# ---------------------------------------------------------------------------

def _vram_callback(frame: StereoInertialFrame, system: MACVO_IMU, pb: ColoredTqdm):
    if torch.cuda.is_available():
        mem_gb = round(torch.cuda.memory_reserved(0) / 1e9, 3)
        mem_str = f"{mem_gb} GB"
    else:
        mem_str = "N/A"
    pb.set_description(desc=f"{system.graph}, VRAM={mem_str}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="MAC-VO + IMU runner for KITTI sequences"
    )
    p.add_argument(
        "--odom", type=str, required=True,
        help="Path to odometry config YAML (e.g. Config/Experiment/MACVO/MACVO_Fast_IMU.yaml). "
             "If the config contains a 'Data:' section, --data can be omitted.",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Path to sequence config YAML with type=KITTI_IMU. "
             "If omitted, the Data: section embedded in --odom is used.",
    )
    p.add_argument("--seq_from", type=int, default=0,
                   help="Start frame index (inclusive, default 0)")
    p.add_argument("--seq_to",   type=int, default=None,
                   help="End frame index (exclusive, default: full sequence)")
    p.add_argument("--resultRoot", type=str, default="./Results",
                   help="Root directory for experiment sandboxes")
    p.add_argument("--preload", action="store_true",
                   help="Preload entire sequence into RAM before running")
    p.add_argument("--autoremove", action="store_true",
                   help="Delete sandbox on exit (useful for debugging)")
    p.add_argument("--noeval", action="store_true",
                   help="Skip ATE/RTE/ROE evaluation after the run")
    p.add_argument("--timing", action="store_true",
                   help="Enable Utility.Timer profiling")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_args()

    # ── Config loading ─────────────────────────────────────────────────────
    cfg,     cfg_dict     = load_config(Path(args.odom))
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

    # ── Sandbox setup ──────────────────────────────────────────────────────
    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    if args.autoremove:
        exp_space.set_autoremove()
    exp_space.config = {
        "Project"  : project_name,
        "Odometry" : odomcfg_dict,
        "Data"     : {"args": datacfg_dict, "start_idx": args.seq_from, "end_idx": args.seq_to},
    }

    Timer.setup(active=args.timing)

    # ── Data source ────────────────────────────────────────────────────────
    # Auto-upgrade plain KITTI → KITTI_IMU so the user can pass KITTI_K07.yaml
    # directly without needing a separate _IMU variant of the sequence config.
    seq_type = datacfg.type
    if seq_type == "KITTI":
        seq_type = "KITTI_IMU"
        Logger.write("info", "Sequence type 'KITTI' upgraded to 'KITTI_IMU' for IMU support.")
    raw_seq = SequenceBase[StereoInertialFrame].instantiate(
        seq_type, datacfg.args
    ).clip(args.seq_from, args.seq_to)

    sequence = smart_transform(raw_seq, cfg.Preprocess)

    if args.preload:
        Logger.write("warn", "Preloading a full KITTI sequence may use >20 GB RAM. "
                             "Omit --preload for long sequences.")
        sequence = sequence.preload()

    # ── System ─────────────────────────────────────────────────────────────
    system = MACVO_IMU.from_config(asNamespace(exp_space.config))

    def on_frame_finished(frame: StereoInertialFrame, sys: MACVO_IMU, pb: ColoredTqdm):
        _vram_callback(frame, sys, pb)

    system.receive_frames(sequence, exp_space, on_frame_finished=on_frame_finished)

    # ── Timing report ──────────────────────────────────────────────────────
    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    # ── Evaluation ─────────────────────────────────────────────────────────
    if not args.noeval:
        header, result = EvaluateSequences([str(exp_space.folder)], correct_scale=False)
        print_as_table(header, result)
