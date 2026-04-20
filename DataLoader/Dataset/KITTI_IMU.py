"""
KITTI Stereo + IMU DataLoader
=============================
Extends the existing KITTI stereo loader to also emit IMU measurements
(from the OXTS system) for each stereo frame.

Dataset layout expected
-----------------------
sequences/<seq>/
├── image_2/                  ← left stereo (10 Hz)
├── image_3/                  ← right stereo (10 Hz)
├── calib.txt
├── calib_imu_to_velo.txt
├── calib_velo_to_cam.txt
├── oxts/
│   ├── data/
│   │   ├── 0000000000.txt    ← one file per IMU sample (~100 Hz)
│   │   └── …
│   └── timestamps.txt        ← one timestamp per IMU sample
└── times.txt                 ← one timestamp per stereo frame (seconds from 0)

OXTS column layout (0-indexed)
-------------------------------
11–13 → ax, ay, az  [m s⁻²]  (accelerometer, raw – gravity not removed)
17–19 → wx, wy, wz  [rad s⁻¹] (gyroscope)

Coordinate-system notes
------------------------
* OXTS frame: x=forward, y=left, z=up  →  gravity = [0, 0, −9.81]
* Camera EDN frame: x=right, y=down, z=forward
* Internal NED frame used by MACVO: x=north, y=east, z=down
* T_imu_to_cam converts raw OXTS vectors to camera (EDN) frame.
  In IMUData, T_BS is set to T_imu_to_cam so downstream code can
  rotate measurements into the body frame as needed.
"""

import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ..Interface import StereoData, IMUData, StereoInertialFrame
from ..SequenceBase import SequenceBase
from .KITTI import KITTIMonocularDataset, loadKITTIGTPoses, NED2EDN

# Gravity constant used throughout (m s⁻²)
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_rt(path: Path, r_key: str = "R:", t_key: str = "T:") -> np.ndarray:
    """Read a KITTI-style calibration file and return the 4×4 SE(3) matrix."""
    R = np.eye(3)
    t = np.zeros(3)
    with open(path) as f:
        for line in f:
            if line.startswith(r_key):
                R = np.array(line[len(r_key):].split(), dtype=float).reshape(3, 3)
            elif line.startswith(t_key):
                t = np.array(line[len(t_key):].split(), dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def load_T_imu_to_cam(seq_root: Path) -> np.ndarray:
    """
    Compute T_imu_to_cam = T_velo_to_cam @ T_imu_to_velo.

    Both calibration files are read from *seq_root*.
    """
    T_imu_to_velo = _load_rt(seq_root / "calib_imu_to_velo.txt")
    T_velo_to_cam = _load_rt(seq_root / "calib_velo_to_cam.txt")
    return T_velo_to_cam @ T_imu_to_velo


# ---------------------------------------------------------------------------
# OXTS / IMU timestamp parsing
# ---------------------------------------------------------------------------

def _parse_oxts_timestamps(timestamps_file: Path) -> np.ndarray:
    """
    Parse OXTS timestamps and return them as nanoseconds **relative to
    the first sample** so they align with stereo *times.txt* (which also
    starts at 0).

    Supports two formats:
    • datetime  – "YYYY-MM-DD HH:MM:SS.nnnnnnnnn"
    • float sec – "0.00000000"
    """
    raw: list[int] = []
    with open(timestamps_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                t_str = parts[1] if len(parts) >= 2 else parts[0]
                h_s, m_s, s_full = t_str.split(":")
                if "." in s_full:
                    s_s, frac = s_full.split(".")
                    ns_frac = int(frac.ljust(9, "0")[:9])
                else:
                    s_s, ns_frac = s_full, 0
                ns = (int(h_s) * 3600 + int(m_s) * 60 + int(s_s)) * 1_000_000_000 + ns_frac
            except Exception:
                ns = int(float(line) * 1_000_000_000)
            raw.append(ns)

    arr = np.array(raw, dtype=np.int64)
    arr -= arr[0]          # make relative to first sample
    return arr


# ---------------------------------------------------------------------------
# OXTS measurement loading
# ---------------------------------------------------------------------------

def _load_oxts_measurements(oxts_dir: Path) -> np.ndarray:
    """
    Load all OXTS .txt files and stack them into an (N_imu, 30) array.
    """
    files = sorted((oxts_dir / "data").glob("*.txt"))
    rows = [np.loadtxt(str(f)) for f in files]
    return np.stack(rows, axis=0)  # (N_imu, 30)


# ---------------------------------------------------------------------------
# Main sequence class
# ---------------------------------------------------------------------------

class KITTI_IMU_Sequence(SequenceBase[StereoInertialFrame]):
    """
    KITTI stereo + OXTS IMU sequence.

    For each stereo frame *i* the loader collects every IMU sample whose
    timestamp falls in the half-open interval (t_{i-1}, t_i].  Frame 0
    gets the samples in (t_0 − 100 ms, t_0].
    """

    @classmethod
    def name(cls) -> str:
        return "KITTI_IMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any], **_):
        cfg = self.config_dict2ns(config)

        self.root          = Path(cfg.root)
        self.sequence_name = self.root.name

        # ── Stereo images ──────────────────────────────────────────────────
        self.imageL = KITTIMonocularDataset(Path(self.root, "image_2"))
        self.imageR = KITTIMonocularDataset(Path(self.root, "image_3"))
        assert len(self.imageL) == len(self.imageR)

        # Stereo timestamps in ns (relative to t=0, from times.txt)
        self.stereo_ts_ns: np.ndarray = self.imageL.cam_timestamps   # (N_stereo,)

        # ── Ground-truth poses ─────────────────────────────────────────────
        if cfg.gt_pose:
            self.gtPose_data = loadKITTIGTPoses(
                Path(self.root.parent.parent, "poses", self.sequence_name + ".txt")
            )
        else:
            self.gtPose_data = None

        # ── Camera intrinsics & extrinsics (same as KITTI.py) ──────────────
        with open(Path(self.root, "calib.txt")) as f:
            lines = f.read().strip().splitlines()

        P2 = np.array(lines[2][4:].split(), dtype=float).reshape(3, 4)
        self.cam2_K_np, self.cam2_R, self.cam2_t, *_ = cv2.decomposeProjectionMatrix(P2)
        self.cam2_t = self.cam2_t[:3] / self.cam2_t[3]
        self.cam2_K = torch.tensor(self.cam2_K_np, dtype=torch.float32).unsqueeze(0)

        P3 = np.array(lines[3][4:].split(), dtype=float).reshape(3, 4)
        _, _, cam3_t, *_ = cv2.decomposeProjectionMatrix(P3)
        cam3_t = cam3_t[:3] / cam3_t[3]

        self.baseline = float(np.linalg.norm(self.cam2_t - cam3_t))

        T_BS_np              = np.eye(4)[np.newaxis]
        T_BS_np[0, :3, :3]  = self.cam2_R
        T_BS_np[0, :3,  3]  = self.cam2_t[..., 0]
        self.cam_T_BS = (
            pp.from_matrix(torch.tensor(T_BS_np, dtype=torch.float32), pp.SE3_type)
            @ NED2EDN.unsqueeze(0)
        )

        # ── IMU calibration ────────────────────────────────────────────────
        T_imu_to_cam_np     = load_T_imu_to_cam(self.root)         # (4, 4)
        T_imu_to_cam_t      = torch.tensor(
            T_imu_to_cam_np[np.newaxis], dtype=torch.float32
        )
        # T_BS for IMU: transform from IMU sensor frame to camera body frame
        self.imu_T_BS = pp.from_matrix(T_imu_to_cam_t, pp.SE3_type)

        # ── OXTS raw measurements and timestamps ───────────────────────────
        oxts_dir              = self.root / "oxts"
        self.imu_meas         = _load_oxts_measurements(oxts_dir)     # (N_imu, 30)
        self.imu_ts_ns        = _parse_oxts_timestamps(               # (N_imu,)
            oxts_dir / "timestamps.txt"
        )

        # Gravity constant (OXTS frame: x-fwd, y-left, z-up → g points -z)
        # After T_imu_to_cam the integration frame is camera-EDN; downstream
        # code that needs NED gravity should apply EDN2NED.
        self.gravity = GRAVITY

        super().__init__(len(self.imageL))

    # -----------------------------------------------------------------------
    # IMU slicing helper
    # -----------------------------------------------------------------------

    def _imu_between(
        self, t_start_ns: int, t_end_ns: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return IMU samples with timestamps in (t_start_ns, t_end_ns].

        Returns
        -------
        acc      : (N, 3) float32  – accelerometer readings [m s⁻²]
        gyro     : (N, 3) float32  – gyroscope readings [rad s⁻¹]
        time_ns  : (N,)   int64    – timestamps relative to sequence start
        """
        mask = (self.imu_ts_ns > t_start_ns) & (self.imu_ts_ns <= t_end_ns)
        subset = self.imu_meas[mask]           # (N, 30)
        ts     = self.imu_ts_ns[mask]          # (N,)

        if len(subset) == 0:
            # No IMU data in window — emit a single zero measurement
            mid = (t_start_ns + t_end_ns) // 2
            return (
                torch.zeros(1, 3, dtype=torch.float32),
                torch.zeros(1, 3, dtype=torch.float32),
                torch.tensor([mid], dtype=torch.int64),
            )

        acc  = torch.tensor(subset[:, 11:14], dtype=torch.float32)  # ax ay az
        gyro = torch.tensor(subset[:, 17:20], dtype=torch.float32)  # wx wy wz
        time = torch.tensor(ts,               dtype=torch.int64)
        return acc, gyro, time

    # -----------------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------------

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index  = self.get_index(local_index)
        imageL = self.imageL[index]

        t_cur_ns  = int(self.stereo_ts_ns[index])
        t_prev_ns = int(self.stereo_ts_ns[index - 1]) if index > 0 else t_cur_ns - 100_000_000

        acc, gyro, imu_ts = self._imu_between(t_prev_ns, t_cur_ns)

        imu_data = IMUData(
            T_BS    = cast(pp.LieTensor, self.imu_T_BS),
            time_ns = imu_ts.unsqueeze(0),        # (1, N)
            gravity = [self.gravity],
            acc     = acc.unsqueeze(0),            # (1, N, 3)
            gyro    = gyro.unsqueeze(0),           # (1, N, 3)
        )

        gt_pose = (
            None
            if self.gtPose_data is None
            else cast(pp.LieTensor, self.gtPose_data[index].unsqueeze(0))
        )

        return StereoInertialFrame(
            stereo = StereoData(
                T_BS     = cast(pp.LieTensor, self.cam_T_BS),
                K        = self.cam2_K,
                baseline = torch.tensor([self.baseline]),
                time_ns  = [t_cur_ns],
                height   = imageL.size(2),
                width    = imageL.size(3),
                imageL   = imageL,
                imageR   = self.imageR[index],
            ),
            imu     = imu_data,
            idx     = [local_index],
            time_ns = [t_cur_ns],
            gt_pose = gt_pose,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"   : lambda v: isinstance(v, str),
            "gt_pose": lambda b: isinstance(b, bool),
        })
