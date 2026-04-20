"""
IMU Preintegrator
=================
Wraps PyPose's ``pp.module.IMUPreintegrator`` to compute preintegrated
IMU measurements between consecutive stereo keyframes.

The preintegrated quantities (ΔR, Δv, Δp) are expressed relative to the
initial IMU pose at the start of the integration window, making them
independent of the absolute world-frame trajectory.  They serve as
constraints in the sliding-window PVGO inside ``MACVO_IMU``.

Coordinate assumptions
----------------------
* Measurements (acc, gyro) are in the **camera EDN frame** after being
  rotated by R_imu_to_cam (rotation part of T_imu_to_cam).
* Gravity in camera EDN frame: g = [0, +9.81, 0]  (y-axis points down).
  Set ``gravity_vec`` accordingly when constructing the integrator.
* If you prefer to work in NED before calling, pass g = [0, 0, +9.81].
"""

from __future__ import annotations

import torch
import pypose as pp
import pypose.module as pm
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class PreintegratedIMU:
    """
    Preintegrated IMU measurement between two consecutive stereo frames.

    All tensors have a leading batch dimension B=1 to match the rest of
    the MACVO pipeline.
    """
    # Preintegrated relative rotation  (pp.SO3, shape (1,))
    delta_R: pp.LieTensor
    # Preintegrated velocity increment in *initial* body frame  (1, 3)
    delta_v: torch.Tensor
    # Preintegrated position increment in *initial* body frame  (1, 3)
    delta_p: torch.Tensor
    # Total time of the integration window  [seconds]
    dt_total: float
    # (Optional) 9×9 covariance of [dR_vec, dv, dp] at final step
    cov: Optional[torch.Tensor] = field(default=None)


# ---------------------------------------------------------------------------
# Preintegrator
# ---------------------------------------------------------------------------

class IMUPreintegrator:
    """
    Computes preintegrated IMU measurements for the sliding-window PVGO.

    Parameters
    ----------
    gravity : float
        Scalar gravity magnitude [m s⁻²].  PyPose's IMUPreintegrator places
        this along the +z axis internally (NED convention: z-down).
        Measurements must be in NED before calling ``preintegrate``.
    acc_noise_std  : float   σ of accelerometer white noise [m s⁻²]
    gyro_noise_std : float   σ of gyroscope white noise [rad s⁻¹]
    prop_cov : bool   Whether to propagate covariance.
    """

    def __init__(
        self,
        gravity:        float = 9.81,
        acc_noise_std:  float = 0.02,
        gyro_noise_std: float = 0.002,
        prop_cov:       bool  = False,
    ) -> None:
        self.gravity       = gravity           # scalar for pp.module.IMUPreintegrator
        self.prop_cov      = prop_cov
        self.acc_noise_var = acc_noise_std  ** 2
        self.gyro_noise_var= gyro_noise_std ** 2

    # ------------------------------------------------------------------
    def preintegrate(
        self,
        acc:     torch.Tensor,   # (1, N, 3) – accelerometer readings
        gyro:    torch.Tensor,   # (1, N, 3) – gyroscope readings
        time_ns: torch.Tensor,   # (1, N)    – timestamps in nanoseconds
    ) -> PreintegratedIMU:
        """
        Preintegrate *N* IMU measurements and return the relative
        (ΔR, Δv, Δp) expressed in the frame at the **start** of the window.

        Parameters
        ----------
        acc      : (1, N, 3)  Accelerometer readings in integration frame
        gyro     : (1, N, 3)  Gyroscope readings in integration frame
        time_ns  : (1, N)     Timestamps in nanoseconds

        Returns
        -------
        PreintegratedIMU
        """
        B, N, _ = acc.shape
        device   = acc.device

        # ── time deltas ────────────────────────────────────────────────────
        if N > 1:
            dt_ns    = (time_ns[:, 1:] - time_ns[:, :-1]).float()   # (B, N-1)
            dt       = (dt_ns / 1e9).unsqueeze(-1)                   # (B, N-1, 1) seconds
            dt_total = float((time_ns[0, -1] - time_ns[0, 0]).item()) / 1e9
        else:
            dt       = torch.full((B, 1, 1), 0.01, dtype=torch.float32, device=device)
            dt_total = 0.01

        # ── trivial case ───────────────────────────────────────────────────
        if N <= 1:
            return PreintegratedIMU(
                delta_R  = pp.identity_SO3(B).to(device),
                delta_v  = torch.zeros(B, 3, device=device),
                delta_p  = torch.zeros(B, 3, device=device),
                dt_total = dt_total,
                cov      = None,
            )

        # ── measurements for integration (N−1 intervals) ──────────────────
        acc_in  = acc[:, :-1, :]    # (B, N-1, 3)
        gyro_in = gyro[:, :-1, :]   # (B, N-1, 3)

        # Identity initial state (preintegration gives relative measurement)
        init_pos = torch.zeros(B, 3, device=device)
        init_rot = pp.identity_SO3(B).to(device)
        init_vel = torch.zeros(B, 3, device=device)

        # pp.module.IMUPreintegrator takes gravity as a scalar float.
        # It internally stores torch.tensor([0, 0, gravity]) – NED z-down.
        integrator = pm.IMUPreintegrator(
            pos      = init_pos,
            rot      = init_rot,
            vel      = init_vel,
            gravity  = float(self.gravity),   # scalar!
            prop_cov = self.prop_cov,
            reset    = True,
        ).to(device)

        # Noise variances passed as scalars (pp accepts float or per-axis tensor)
        with torch.no_grad():
            state = integrator(
                dt       = dt,
                gyro     = gyro_in,
                acc      = acc_in,
                gyro_cov = self.gyro_noise_var if self.prop_cov else None,
                acc_cov  = self.acc_noise_var  if self.prop_cov else None,
            )

        # state['pos'] shape: (B, N,   3)  – includes initial 0 + N-1 steps
        # state['rot'] shape: (B, N)       – pp.SO3 LieTensor
        # state['vel'] shape: (B, N,   3)
        # state['cov'] shape: (B, N,   9, 9) if prop_cov

        delta_R = state["rot"][:, -1]    # pp.SO3, shape (B,)
        delta_v = state["vel"][:, -1]    # (B, 3)
        delta_p = state["pos"][:, -1]    # (B, 3)

        cov = None
        if self.prop_cov and "cov" in state:
            cov = state["cov"][:, -1]    # (B, 9, 9)

        return PreintegratedIMU(
            delta_R  = delta_R,
            delta_v  = delta_v,
            delta_p  = delta_p,
            dt_total = dt_total,
            cov      = cov,
        )
