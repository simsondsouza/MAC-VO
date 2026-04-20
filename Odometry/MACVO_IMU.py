"""
MAC-VO + IMU  (Odometry Module)
================================
Extends ``MACVO`` with IMU preintegration and a sliding-window
Pose-Velocity-Gyro Optimizer (PVGO) following the iSLAM paper:

    "iSLAM: Imperative SLAM" – Minghan Zhu et al., RA-L 2024

Architecture
------------
* **Tier 1 (local)** – Standard MAC-VO two-frame visual tracking, unchanged.
* **Tier 1.5 (IMU layer)** – After each visual step:
    1. Preintegrate OXTS measurements for the current frame window.
    2. Maintain a sliding window of size ``W`` storing poses, velocities
       and preintegrated IMU.
    3. When the window is full, run a short PyPose Levenberg-Marquardt
       solve that minimises four residuals (§ Cost below).

Cost function
-------------
    L = λ₁·L_MACVO  +  λ₂·L_rot  +  λ₃·L_cross  +  λ₄·L_deltav  +  λ₅·L_bias

where for each consecutive pair (k, k+1) in the window:

    L_rot   = ‖Log(ΔR⁻¹ · Rₖᵀ · Rₖ₊₁)‖²            (SO3 rotation)
    L_cross = ‖Rₖᵀ(tₖ₊₁ − tₖ − vₖ·Δt) − Δp − ½gΔt²‖²   (position)
    L_deltav= ‖vₖ₊₁ − vₖ − g·Δt − Rₖ·Δv‖²            (velocity)
    L_MACVO = ‖Log(T_vis⁻¹ · Rₖᵀ · Rₖ₊₁)‖²  for the rotation part
              + ‖t_vis − Rₖᵀ(tₖ₊₁ − tₖ)‖²    for the translation part

State per node: (Rₖ ∈ SO3,  tₖ ∈ ℝ³,  vₖ ∈ ℝ³,  bₐₖ ∈ ℝ³,  bᵍₖ ∈ ℝ³)
where bₐ is the accelerometer bias and bᵍ is the gyroscope bias,
both modelled as a random walk: L_bias = λ_bias·‖bₐₖ₊₁ − bₐₖ‖² + λ_bias·‖bᵍₖ₊₁ − bᵍₖ‖²

Gravity convention
------------------
Internally MACVO uses NED (North-East-Down) world poses.
Gravity vector in NED: g = [0, 0, +9.81].
The IMU measurements must be rotated into NED before they reach the PVGO.
The rotation R_imu_to_ned = R_imu_to_cam · EDN2NED is applied inside
``_rotate_imu_to_ned`` before preintegration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppopt
from pypose.optim import LM as ppLM
from pypose.optim.scheduler import StopOnPlateau as ppStopOnPlateau
from types import SimpleNamespace
from typing import Callable

import Module
from DataLoader import StereoInertialFrame
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer

from .MACVO import MACVO
from .IMUPreintegrator import IMUPreintegrator, PreintegratedIMU

def _mat3_to_SO3(R: torch.Tensor) -> pp.LieTensor:
    """Convert a (3,3) rotation matrix to a pp.SO3 LieTensor (shape ())."""
    H = torch.eye(4, dtype=R.dtype, device=R.device)
    H[:3, :3] = R
    return pp.from_matrix(H, pp.SE3_type).rotation()


# ---------------------------------------------------------------------------
# Sliding-window cost model (PyPose-compatible nn.Module)
# ---------------------------------------------------------------------------

class _SlidingWindowCost(nn.Module):
    """
    PyPose-compatible cost module for the sliding-window PVGO.

    Parameters
    ----------
    init_poses   : list of pp.LieTensor (SE3, shape (7,)) – initial poses
    init_vels    : list of torch.Tensor (3,) – initial velocities
    imu_factors  : list of PreintegratedIMU – one per consecutive pair, deltas
                   in camera (EDN) body frame, gravity already removed by PyPose
    vis_rel_poses: list of pp.LieTensor (SE3, shape (7,)) – visual relative
                   poses (from visual optimizer), one per consecutive pair
    lambdas      : tuple (λ_macvo, λ_rot, λ_cross, λ_deltav, λ_bias)
    init_bias_a  : list of torch.Tensor (3,) – initial accelerometer biases (W nodes)
    init_bias_g  : list of torch.Tensor (3,) – initial gyroscope biases (W nodes)
    """

    def __init__(
        self,
        init_poses:    list[pp.LieTensor],
        init_vels:     list[torch.Tensor],
        imu_factors:   list[PreintegratedIMU],
        vis_rel_poses: list[pp.LieTensor],
        lambdas:       tuple[float, float, float, float, float],
        init_bias_a:   list[torch.Tensor] | None = None,
        init_bias_g:   list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        W = len(init_poses)
        assert len(init_vels)     == W
        assert len(imu_factors)   == W - 1
        assert len(vis_rel_poses) == W - 1

        # ── Frame 0: hard-frozen constant (not a variable) ────────────────
        # Removing frame 0 from the optimisation variables eliminates the
        # translational/rotational null space without adding any residual.
        # A soft anchor (lambda * residual) distorts poses even with all
        # IMU lambdas = 0; a hard freeze has zero effect on the other frames
        # when all IMU terms are disabled.
        self.register_buffer("R0", init_poses[0].rotation().tensor())  # (4,) SO3
        self.register_buffer("t0", init_poses[0].translation())         # (3,)

        # ── Optimisable parameters: frames 1 .. W-1 ───────────────────────
        rot_data = torch.stack([p.rotation().tensor() for p in init_poses[1:]])  # (W-1, 4)
        self.rotations    = pp.Parameter(pp.SO3(rot_data))                # (W-1,) SO3
        self.translations = nn.Parameter(
            torch.stack([p.translation() for p in init_poses[1:]])        # (W-1, 3)
        )
        self.velocities   = nn.Parameter(
            torch.stack(init_vels)                                         # (W, 3)
        )

        # ── Bias state (one vector per node, all W frames) ────────────────
        # First-order correction to preintegrated IMU measurements.
        # Accelerometer bias bₐ: corrects Δp and Δv.
        # Gyroscope bias bᵍ: corrects ΔR.
        if init_bias_a is None:
            init_bias_a = [torch.zeros(3)] * W
        if init_bias_g is None:
            init_bias_g = [torch.zeros(3)] * W
        assert len(init_bias_a) == W
        assert len(init_bias_g) == W
        self.bias_a = nn.Parameter(torch.stack(init_bias_a))   # (W, 3)
        self.bias_g = nn.Parameter(torch.stack(init_bias_g))   # (W, 3)

        # ── Fixed buffers ──────────────────────────────────────────────────
        for i, f in enumerate(imu_factors):
            self.register_buffer(f"dR_{i}", f.delta_R.tensor())      # (1, 4) quaternion
            self.register_buffer(f"dv_{i}", f.delta_v.squeeze(0))    # (3,)
            self.register_buffer(f"dp_{i}", f.delta_p.squeeze(0))    # (3,)
            self.register_buffer(f"dt_{i}", torch.tensor(f.dt_total))

        for i, vp in enumerate(vis_rel_poses):
            self.register_buffer(f"vR_{i}", vp.rotation().tensor())  # (4,)
            self.register_buffer(f"vt_{i}", vp.translation())         # (3,)

        self.W       = W
        self.lambdas = lambdas

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:           # → stacked residuals (M,)
        lmv, lr, lc, ldv, lb = self.lambdas
        residuals: list[torch.Tensor] = []

        for i in range(self.W - 1):
            # Frame k: index 0 uses the hard-frozen constant; indices 1..W-1
            # use the optimisable parameters at position i-1.
            if i == 0:
                R_k = pp.SO3(self.R0)   # frozen – no gradient
                t_k = self.t0           # frozen – no gradient
            else:
                R_k = self.rotations[i - 1]
                t_k = self.translations[i - 1]

            # Frame k+1 is always optimisable (parameter index i).
            R_kp1 = self.rotations[i]
            t_kp1 = self.translations[i]
            v_k   = self.velocities[i]            # (3,)
            v_kp1 = self.velocities[i + 1]        # (3,)

            # Biases for frame k and k+1
            b_a_k   = self.bias_a[i]              # (3,) accelerometer bias at k
            b_g_k   = self.bias_g[i]              # (3,) gyroscope bias at k
            b_a_kp1 = self.bias_a[i + 1]          # (3,)
            b_g_kp1 = self.bias_g[i + 1]          # (3,)

            dR   = pp.SO3(getattr(self, f"dR_{i}").squeeze(0))   # SO3
            dv   = getattr(self, f"dv_{i}")                       # (3,)
            dp   = getattr(self, f"dp_{i}")                       # (3,)
            dt   = getattr(self, f"dt_{i}")                       # scalar

            vR   = pp.SO3(getattr(self, f"vR_{i}"))               # SO3
            vt   = getattr(self, f"vt_{i}")                       # (3,)

            # ── L_rot : IMU rotation residual (with gyro bias correction) ─
            # True preintegration: ΔR_true ≈ ΔR_meas · Exp(-Δt · bᵍ)
            # Residual first-order: r_rot ≈ Log(ΔR⁻¹ · Rₖᵀ · Rₖ₊₁) + Δt · bᵍₖ
            r_rot = (dR.Inv() @ R_k.Inv() @ R_kp1).Log() + dt * b_g_k   # (3,)

            # ── L_cross : position residual (with acc bias correction) ────
            # True Δp_true ≈ Δp_meas - (Δt²/2) · bₐ
            # Residual: r_cross = Rₖᵀ(tₖ₊₁ − tₖ − vₖΔt) − Δp_meas + (Δt²/2)·bₐₖ
            r_cross = (
                R_k.Inv().Act(t_kp1 - t_k - v_k * dt) - dp
                + (dt ** 2 / 2) * b_a_k
            )                                                       # (3,)

            # ── L_deltav : velocity residual (with acc bias correction) ───
            # Bias bₐ is in camera body frame; correction kept in body frame
            # (not rotated to world) for simplicity.
            r_deltav = v_kp1 - v_k - R_k.Act(dv) + dt * b_a_k           # (3,)

            # ── L_MACVO : visual relative-pose residual ───────────────────
            rel_R_est = R_k.Inv() @ R_kp1
            rel_t_est = R_k.Inv().Act(t_kp1 - t_k)
            r_vis_rot = (vR.Inv() @ rel_R_est).Log()               # (3,)
            r_vis_t   = rel_t_est - vt                              # (3,)

            # ── Bias random walk + magnitude regularisation ───────────────
            # Random walk: penalises bias *change* between consecutive nodes.
            # Magnitude term: penalises large absolute bias values, keeping the
            # optimiser away from degenerate (NaN) regions of the cost surface.
            r_bias_a = b_a_kp1 - b_a_k                             # (3,)
            r_bias_g = b_g_kp1 - b_g_k                             # (3,)
            r_mag_a  = b_a_k                                        # (3,) magnitude
            r_mag_g  = b_g_k                                        # (3,) magnitude

            residuals += [
                lmv * r_vis_rot,
                lmv * r_vis_t,
                lr  * r_rot,
                lc  * r_cross,
                ldv * r_deltav,
                lb  * r_bias_a,
                lb  * r_bias_g,
                lb  * r_mag_a,   # magnitude regularisation
                lb  * r_mag_g,
            ]

        return torch.cat(residuals)              # (W-1) × 27 scalars


# ---------------------------------------------------------------------------
# Main MACVO+IMU odometry class
# ---------------------------------------------------------------------------

class MACVO_IMU(MACVO[StereoInertialFrame]):
    """
    MAC-VO augmented with IMU preintegration and sliding-window PVGO.

    Additional constructor parameters (passed via config.Odometry.args)
    -------------------------------------------------------------------
    window_size  : int   Sliding-window length W (default 6).
    lambda_macvo : float Weight for visual residual (default 1.0).
    lambda_rot   : float Weight for IMU rotation residual (default 1.0).
    lambda_cross : float Weight for translation-velocity residual (default 0.1).
    lambda_deltav: float Weight for velocity residual (default 0.1).
    lambda_bias  : float Weight for bias random-walk residual (default 10.0).
    lm_steps     : int   Max LM iterations per window (default 30).
    gravity      : float Scalar gravity [m s⁻²] (default 9.81).
    """

    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile, mapping,
        frontend        : Module.IFrontend,
        motion_model    : Module.IMotionModel,
        kp_selector     : Module.IKeypointSelector,
        map_selector    : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.ICovariance2to3,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector,
        optimizer       : Module.IOptimizer,
        # IMU-specific
        window_size:   int   = 6,
        lambda_macvo:  float = 5.0,   # visual must strongly dominate (5:1 over IMU)
        lambda_rot:    float = 0.5,
        lambda_cross:  float = 0.1,
        lambda_deltav: float = 0.1,
        lambda_bias:   float = 10.0,
        lm_steps:      int   = 30,
        gravity:       float = 9.81,
        **extra,
    ) -> None:
        super().__init__(
            device=device, num_point=num_point, edgewidth=edgewidth,
            match_cov_default=match_cov_default, profile=profile,
            mapping=mapping, frontend=frontend, motion_model=motion_model,
            kp_selector=kp_selector, map_selector=map_selector,
            obs_filter=obs_filter, obs_covmodel=obs_covmodel,
            post_process=post_process, kf_selector=kf_selector,
            optimizer=optimizer, **extra,
        )

        self.window_size  = window_size
        self.lambdas      = (lambda_macvo, lambda_rot, lambda_cross, lambda_deltav, lambda_bias)
        self.lm_steps     = lm_steps

        # IMU preintegrator in OXTS body frame (z=up, gravity=+9.81 correct for z-up).
        # PyPose gravity=+9.81 removes gravity along +z for z-up sensors (OXTS convention).
        self.imu_preint = IMUPreintegrator(
            gravity        = gravity,  # use config value, not hardcoded
            acc_noise_std  = 0.02,
            gyro_noise_std = 0.002,
            prop_cov       = False,
        )

        # R_imu_to_cam cached (set on first frame from T_BS calibration)
        self._R_imu_to_cam: torch.Tensor | None = None

        # Sliding-window buffers (filled after each keyframe pair)
        self._win_poses:     list[pp.LieTensor]     = []   # SE3 (7,) NED
        self._win_vels:      list[torch.Tensor]      = []   # (3,) NED
        self._win_imu:       list[PreintegratedIMU]  = []
        self._win_vis_rel:   list[pp.LieTensor]      = []   # visual relative SE3
        self._win_frame_idx: list[int]               = []   # frame indices in map
        # Bias state carried across windows (last committed PVGO bias estimates)
        self._win_bias_a:    list[torch.Tensor]      = []   # (3,) acc bias per node
        self._win_bias_g:    list[torch.Tensor]      = []   # (3,) gyro bias per node

        # PVGO trajectory: frame_idx -> (7,) PVGO-optimised SE3 [tx,ty,tz,qx,qy,qz,qw].
        # These are absolute poses derived from pre-PGO visual estimates.
        self._pvgo_trajectory: dict[int, torch.Tensor] = {}
        # Pre-PGO visual pose at commit time: frame_idx -> pp.SE3 (absolute, pre-PGO).
        # Retained for debugging; no longer used in get_map() since we write PVGO poses directly.
        self._pvgo_vis_pre: dict[int, pp.LieTensor] = {}

        # Callbacks fired after each successful PVGO commit.
        # Signature: callback(fidx: int, pvgo_se3: pp.LieTensor) -> None
        # Used by MACSLAM to feed Tier-1 (PVGO) output into Tier-2 (global PGO).
        self._pvgo_callbacks: list = []

    # ------------------------------------------------------------------
    # PVGO commit callbacks
    # ------------------------------------------------------------------

    def register_on_pvgo_commit(self, callback) -> None:
        """
        Register a callback invoked after each successful PVGO commit.

        Signature: ``callback(fidx: int, pvgo_se3: pp.LieTensor) -> None``

        Used by MACSLAM so that Tier-2 (global LC PGO) receives
        Tier-1 (PVGO) corrected absolute poses as sequential edge inputs,
        ensuring Tier 1 feeds into Tier 2 rather than both running
        independently on top of raw visual poses.
        """
        self._pvgo_callbacks.append(callback)

    # ------------------------------------------------------------------
    # IMU calibration (lazy init from first frame's T_BS)
    # ------------------------------------------------------------------

    def _get_R_imu_to_cam(self, imu_T_BS: pp.LieTensor) -> torch.Tensor:
        """Return cached (3,3) rotation matrix from OXTS/IMU frame to camera (EDN)."""
        if self._R_imu_to_cam is None:
            self._R_imu_to_cam = imu_T_BS[0].rotation().matrix().detach()  # (3,3)
        return self._R_imu_to_cam

    def _rotate_preint_to_cam(
        self, preint: PreintegratedIMU, R_ic: torch.Tensor
    ) -> PreintegratedIMU:
        """
        Rotate preintegrated IMU deltas from OXTS body frame to camera (EDN) frame.

        Δv_cam = R_ic @ Δv_oxts
        Δp_cam = R_ic @ Δp_oxts
        ΔR_cam = R_ic ⊗ ΔR_oxts ⊗ R_ic⁻¹  (change-of-reference-frame for relative rotation)
        """
        R_ic_so3 = _mat3_to_SO3(R_ic)                               # SO3 ()

        delta_v_cam = preint.delta_v @ R_ic.T                        # (1,3)
        delta_p_cam = preint.delta_p @ R_ic.T                        # (1,3)

        # Conjugate ΔR: ΔR_cam = R_ic ⊗ ΔR_oxts ⊗ R_ic⁻¹
        dR_oxts = preint.delta_R                                      # SO3 (1,)
        dR_cam  = R_ic_so3.unsqueeze(0) @ dR_oxts @ R_ic_so3.Inv().unsqueeze(0)

        return PreintegratedIMU(
            delta_R  = dR_cam,
            delta_v  = delta_v_cam,
            delta_p  = delta_p_cam,
            dt_total = preint.dt_total,
            cov      = preint.cov,
        )

    # ------------------------------------------------------------------
    # Override run_pair to inject IMU
    # ------------------------------------------------------------------

    @Timer.cpu_timeit("IMU_Odom_Runtime")
    def run_pair(
        self,
        frame0: StereoInertialFrame,
        frame1: StereoInertialFrame,
    ) -> None:
        # ── 1. Preintegrate IMU in OXTS body frame (z=up, gravity=+9.81 correct)
        #       then rotate Δv/Δp/ΔR to camera (EDN) body frame.
        R_ic = self._get_R_imu_to_cam(frame1.imu.T_BS)          # (3,3)
        preint_oxts = self.imu_preint.preintegrate(
            acc     = frame1.imu.acc,                             # raw OXTS, (1,N,3)
            gyro    = frame1.imu.gyro,                            # raw OXTS, (1,N,3)
            time_ns = frame1.imu.time_ns,
        )
        preint = self._rotate_preint_to_cam(preint_oxts, R_ic)   # now in cam frame

        # ── 2. Run standard MAC-VO visual processing ───────────────────────
        prev_map_len = len(self.graph.frames)
        super().run_pair(frame0, frame1)
        new_map_len  = len(self.graph.frames)

        # If super() didn't add a keyframe (non-keyframe path), skip PVGO
        if new_map_len == prev_map_len:
            return

        cur_frame_idx = int(new_map_len - 1)

        # ── 3. Retrieve current pose from graph (motion model initial estimate)
        # Note: super().run_pair() writes the *previous* frame's optimized pose
        # at its start. The current frame's pose here is the motion model guess;
        # the PVGO below will refine it before the next call writes it back.
        cur_pose_se3 = pp.SE3(self.graph.frames.data["pose"][cur_frame_idx].clone())

        # Visual relative pose (for L_MACVO in PVGO)
        if len(self._win_poses) > 0:
            prev_pose_se3 = self._win_poses[-1]
            vis_rel = prev_pose_se3.Inv() @ cur_pose_se3
        else:
            vis_rel = pp.identity_SE3(1).squeeze(0)

        # ── 4. Estimate velocity via finite difference ─────────────────────
        if len(self._win_poses) > 0 and preint.dt_total > 1e-6:
            vel = (
                cur_pose_se3.translation() - self._win_poses[-1].translation()
            ) / preint.dt_total
        else:
            vel = torch.zeros(3)

        # ── 5. Update sliding window ───────────────────────────────────────
        self._win_poses.append(cur_pose_se3.detach())
        self._win_vels.append(vel.detach())
        self._win_frame_idx.append(cur_frame_idx)
        # Carry the last optimised bias estimate forward; zero-initialise for the first frame.
        prev_bias_a = self._win_bias_a[-1].clone() if self._win_bias_a else torch.zeros(3)
        prev_bias_g = self._win_bias_g[-1].clone() if self._win_bias_g else torch.zeros(3)
        self._win_bias_a.append(prev_bias_a)
        self._win_bias_g.append(prev_bias_g)

        if len(self._win_imu) < len(self._win_poses) - 1 or len(self._win_poses) == 1:
            # pad vis_rel list so indices match (no IMU for very first frame)
            pass
        if len(self._win_poses) > 1:
            self._win_imu.append(preint)
            self._win_vis_rel.append(vis_rel.detach())

        # Keep window bounded
        max_buf = self.window_size + 1
        if len(self._win_poses) > max_buf:
            self._win_poses.pop(0)
            self._win_vels.pop(0)
            self._win_frame_idx.pop(0)
            self._win_bias_a.pop(0)
            self._win_bias_g.pop(0)
            self._win_imu.pop(0)
            self._win_vis_rel.pop(0)

        # ── 6. Run PVGO when window is full ───────────────────────────────
        if len(self._win_poses) >= self.window_size:
            self._run_pvgo()

    # ------------------------------------------------------------------
    # Sliding-window PVGO
    # ------------------------------------------------------------------

    def _run_pvgo(self) -> None:
        W = len(self._win_poses)
        if W < 2:
            return

        # Align lengths: poses[0..W-1], imu/vis_rel[0..W-2]
        n_factors = min(len(self._win_imu), W - 1)
        if n_factors < 1:
            return

        poses     = self._win_poses[:n_factors + 1]
        vels      = self._win_vels[:n_factors + 1]
        imu_facs  = self._win_imu[:n_factors]
        vis_rels  = self._win_vis_rel[:n_factors]
        bias_a    = self._win_bias_a[:n_factors + 1]
        bias_g    = self._win_bias_g[:n_factors + 1]

        model = _SlidingWindowCost(
            init_poses    = poses,
            init_vels     = vels,
            imu_factors   = imu_facs,
            vis_rel_poses = vis_rels,
            lambdas       = self.lambdas,
            init_bias_a   = bias_a,
            init_bias_g   = bias_g,
        )

        try:
            optimizer  = ppLM(model, min=1e-4)   # higher min avoids NaN in Jacobian
            scheduler  = ppStopOnPlateau(optimizer, steps=self.lm_steps, patience=3, decreasing=1e-6, verbose=False)
            while scheduler.continual():
                loss = optimizer.step(input=())
                scheduler.step(loss)
        except Exception as e:
            Logger.write("warn", f"PVGO failed with {e}; keeping visual-only poses.")
            return

        # ── Store optimised poses in separate PVGO trajectory (not in visual map)
        # Writing back to the visual map during tracking creates a feedback loop:
        # corrupted PVGO poses affect the motion model → degrade visual tracking.
        # Instead, we accumulate PVGO corrections here and apply them at get_map().
        with torch.no_grad():
            # model.rotations / .translations cover frames 1..W-1 (frame 0 was frozen).
            # Index 0 in these tensors corresponds to window frame 1 (second oldest).
            R_opt = model.rotations.data          # (W-1, 4) SO3 quaternion (raw tensor)
            t_opt = model.translations.data       # (W-1, 3)

            # Guard: skip if optimization produced NaN/Inf
            if torch.isnan(R_opt).any() or torch.isnan(t_opt).any():
                Logger.write("warn", "PVGO write-back skipped: NaN in optimised parameters.")
                return

            # Commit frame 1 (second oldest in window).
            # Frame 0 is hard-frozen to its visual pose.  Frame 1 is the first free
            # variable, optimised relative to the frozen anchor.  Each window slides
            # by 1, so each frame k is committed exactly once as "param index 0"
            # (= window frame 1) with window frame k-1 as the frozen anchor.
            if W < 2:
                return
            param_i     = 0                            # param index for window frame 1
            commit_fidx = self._win_frame_idx[1]       # global frame index
            q_c         = R_opt[param_i] / (R_opt[param_i].norm() + 1e-8)
            pose_commit = torch.cat([t_opt[param_i], q_c], dim=-1).cpu()

            # Sanity check: discard if the pose drifted too far from visual estimate.
            t_vis   = self._win_poses[1].translation().cpu()
            R_vis   = self._win_poses[1].rotation().cpu()      # SO3
            R_pvgo  = pp.SO3(q_c.cpu())
            t_delta = (pose_commit[:3] - t_vis).norm()
            r_delta = (R_vis.Inv() @ R_pvgo).Log().norm()      # rotation error in rad

            if t_delta > 15.0:
                Logger.write("warn", f"PVGO frame-{commit_fidx} rejected "
                             f"(translation moved {t_delta:.2f} m from visual).")
                return
            if r_delta > 1.0:   # ~57° – relaxed to let IMU contribute more
                Logger.write("warn", f"PVGO frame-{commit_fidx} rejected "
                             f"(rotation deviated {r_delta:.3f} rad from visual).")
                return

            self._pvgo_trajectory[commit_fidx] = pose_commit
            # Save the pre-PGO visual pose for debugging / future use.
            self._pvgo_vis_pre[commit_fidx] = self._win_poses[1].detach().clone().cpu()

            # Notify listeners (e.g. MACSLAM) so Tier-2 (global LC PGO) can
            # update its node estimate for this frame with the PVGO-corrected pose.
            # This ensures Tier-2 sequential edges are anchored to Tier-1 output.
            pvgo_se3_notify = pp.SE3(pose_commit.unsqueeze(0).double())
            for cb in self._pvgo_callbacks:
                try:
                    cb(commit_fidx, pvgo_se3_notify)
                except Exception as e:
                    Logger.write("warn", f"PVGO commit callback failed: {e}")

            # Write back ALL optimised biases into the window buffer so every node
            # in the next window warm-starts from the current solution.
            # Physical limits prevent unbounded growth under poor conditioning:
            #   acc bias ≤ 0.5 m/s²  (MEMS IMU spec-sheet upper bound)
            #   gyro bias ≤ 0.05 rad/s  (≈ 3 °/s, conservative MEMS limit)
            MAX_BIAS_A = 0.5   # m/s²
            MAX_BIAS_G = 0.05  # rad/s
            n_nodes = min(model.bias_a.data.shape[0], len(self._win_bias_a))
            for j in range(n_nodes):
                self._win_bias_a[j] = model.bias_a.data[j].detach().cpu().clamp(-MAX_BIAS_A, MAX_BIAS_A)
                self._win_bias_g[j] = model.bias_g.data[j].detach().cpu().clamp(-MAX_BIAS_G, MAX_BIAS_G)

            # Update velocity for the committed frame from the PVGO estimate.
            # model.velocities[1] corresponds to window frame 1 (the committed frame).
            # This is a better estimate than the original finite-difference velocity.
            self._win_vels[1] = model.velocities.data[1].detach().cpu()

            # NOTE: _win_poses always holds VISUAL poses.  Never write PVGO poses
            # back so that vis_rel stays the true visual relative pose for the next
            # window.  Writing back PVGO poses corrupts vis_rel → exponential drift.

    # ------------------------------------------------------------------
    # get_map – apply PVGO-corrected poses to the visual map at the end
    # ------------------------------------------------------------------

    def get_map(self):
        graph = super().get_map()   # post-PGO refined visual poses are now in graph
        if self._pvgo_trajectory:
            n_corrected = len(self._pvgo_trajectory)
            n_total     = len(graph.frames)
            Logger.write("info", f"Applying PVGO corrections to {n_corrected}/{n_total} frames.")
            device = graph.frames.data["pose"].tensor.device
            with torch.no_grad():
                for fidx, pvgo_vec in self._pvgo_trajectory.items():
                    if fidx >= n_total:
                        continue
                    # Write the PVGO absolute pose directly.
                    # This is correct because:
                    #   • pvgo_vec is an absolute SE3 pose in the world frame, not a delta.
                    #   • TwoFrame_PGO only operates on the two most recent frames, so
                    #     older frames are not further refined after PVGO commit.
                    # When MACSLAM wraps this class, its get_map() will apply LC
                    # corrections on top of these PVGO poses (correct two-tier order).
                    graph.frames.data["pose"][fidx] = pvgo_vec.to(device)
        return graph

    # ------------------------------------------------------------------
    # from_config – adds IMU kwargs on top of MACVO defaults
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: SimpleNamespace):
        odomcfg = cfg.Odometry
        Frontend          = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        MotionEstimator   = Module.IMotionModel[StereoInertialFrame].instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector  = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        MappointSelector  = Module.IKeypointSelector.instantiate(odomcfg.mappoint.type, odomcfg.mappoint.args)
        ObsFilter         = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObsCovModel       = Module.ICovariance2to3.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner        = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector  = Module.IKeyframeSelector[StereoInertialFrame].instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer         = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)

        return cls(
            frontend=Frontend, motion_model=MotionEstimator,
            kp_selector=KeypointSelector, map_selector=MappointSelector,
            obs_filter=ObsFilter, obs_covmodel=ObsCovModel,
            post_process=MapRefiner, kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            **vars(odomcfg.args),
        )
