"""
MAC-SLAM  –  MAC-VO + Global Pose Graph with Loop Closure
==========================================================

``MACSLAM`` is the *SystemManager* orchestrator described in the plan.
It wraps the existing ``MACVO`` class (Tier 1 / local) and adds:

* **Place recognition** — MixVPR (WACV 2023) pretrained descriptors encoded
  on every global keyframe and queried against a brute-force cosine-similarity
  database.  Falls back to GeM-pooled ResNet-18 (NetVLADLoopDetector) if
  MixVPR weights are unavailable.
* **Geometric verification** — candidate loop closures are verified using
  ORB keypoints + Essential Matrix RANSAC via ``GeometricVerifier``.  The
  recovered relative pose and inlier-ratio-scaled covariance are pushed
  directly to the global pose graph.
* **Global pose graph** — ``GlobalPGO_Optimizer`` runs asynchronously on a
  background thread with switchable constraints (§ Loop Closure in plan).
* **Correction propagation** — after the global solve converges the
  corrections  δT = T* · T̂⁻¹  are applied rigidly to every keyframe,
  then interpolated to non-keyframes in the local window.

Threading model  (cf.  Fig. "Timing & Threading Model" in plan)
---------------------------------------------------------------
::

    Tier 1 (main)   ──f1──f2──f3──KF──f5──f6──KF+LC──f8──f9──…
                                              │            ↑
                                              ↓ trigger    │ δT
    Tier 2 (bg)     ─────────────────────── [Global LM] ───┘
"""

from __future__ import annotations

import numpy as np
import torch
import pypose as pp
import typing as T
from types import SimpleNamespace

import Module
from DataLoader import StereoFrame
from Module.Map import VisualMap
from Module.LoopClosure.Interface import ILoopClosureDetector, LoopClosureResult
from Module.LoopClosure.NetVLAD_MixVPR import MixVPRLoopDetector #, NetVLADLoopDetector
from Module.Optimization.GlobalPGO.Optimizer import (
    GlobalPGO_Optimizer,
    SequentialEdge,
    LoopClosureEdge,
)
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Math import NormalizeQuat

from .MACVO import MACVO
from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=StereoFrame)

# Detector class lookup for factory dispatch
_LOOP_DETECTOR_REGISTRY: dict[str, type[ILoopClosureDetector]] = {
    "MixVPRLoopDetector":  MixVPRLoopDetector,
    # "NetVLADLoopDetector": NetVLADLoopDetector,
}


class MACSLAM(IOdometry[T_SensorFrame]):
    """
    MAC-SLAM System Manager.

    Parameters
    ----------
    local_vo : MACVO
        Fully-initialised local MAC-VO pipeline.
    loop_detector : ILoopClosureDetector
        VPR encoder + database for candidate retrieval.  Recommended:
        ``MixVPRLoopDetector`` (4096-d MixVPR descriptors + ORB geometric
        verification).  Lightweight fallback: ``NetVLADLoopDetector``.
    global_pgo : GlobalPGO_Optimizer
        Asynchronous global pose-graph backend.
    lc_min_gap : int
        Minimum frame-index gap to accept a LC candidate (avoids matching
        neighbouring frames).
    lc_geometric_threshold : float
        Minimum VPR similarity to proceed with geometric verification.
        Only used by the heuristic fallback path; the ORB-based path in
        ``MixVPRLoopDetector`` uses its own internal thresholds.
    keyframe_freq : int
        Every *keyframe_freq*-th frame is treated as a *global* keyframe
        (encoded for VPR and added to the global pose graph).  This is
        independent of the local ``KeyframeSelector``.
    apply_correction : bool
        If True, δT corrections are propagated after global convergence.
    """

    def __init__(
        self,
        local_vo:                MACVO,
        loop_detector:           ILoopClosureDetector,
        global_pgo:              GlobalPGO_Optimizer,
        lc_min_gap:              int   = 30,
        lc_geometric_threshold:  float = 0.5,
        keyframe_freq:           int   = 5,
        apply_correction:        bool  = True,
    ) -> None:
        super().__init__(profile=local_vo.profile)

        self.local_vo       = local_vo
        self.loop_detector  = loop_detector
        self.global_pgo     = global_pgo

        self.lc_min_gap              = lc_min_gap
        self.lc_geometric_threshold  = lc_geometric_threshold
        self.keyframe_freq           = keyframe_freq
        self.apply_correction        = apply_correction

        # Bookkeeping
        self.global_kf_indices: list[int] = []
        self.frame_count: int = 0
        self.total_lc_detected: int = 0
        self.total_lc_rejected: int = 0  # geometric verification failures

        # Install a hook so that after every local optimisation write-back
        # we automatically push the sequential edge to the global graph.
        self.local_vo.register_on_optimize_finish(self._on_local_optimize_done)

    # ==================================================================
    # Factory
    # ==================================================================

    @classmethod
    def from_config(cls, cfg: SimpleNamespace) -> "MACSLAM":
        """Build MAC-SLAM from a merged config namespace.

        The default loop-closure detector is now ``MixVPRLoopDetector``.
        Set ``cfg.SLAM.loop_closure.type = "NetVLADLoopDetector"`` to
        fall back to the lightweight GeM-ResNet encoder.
        """

        # ── Local MAC-VO ────────────────────────────────────────────────
        local_vo = MACVO[StereoFrame].from_config(cfg)

        # ── SLAM-specific settings (with defaults) ─────────────────────
        slam_cfg = getattr(cfg, "SLAM", None)
        if slam_cfg is None:
            slam_cfg = SimpleNamespace(
                loop_closure = SimpleNamespace(
                    type = "MixVPRLoopDetector",
                    args = SimpleNamespace(
                        device         = "cuda" if torch.cuda.is_available() else "cpu",
                        threshold      = 0.70,
                        top_k          = 3,
                        min_gap        = 50,
                        # MixVPR-specific
                        weight_path    = None,   # auto-discovers Model/*.ckpt
                        # Geometric verifier tunables
                        orb_features              = 1000,
                        match_ratio               = 0.75,
                        ransac_threshold           = 1.0,
                        geometric_min_inliers      = 30,
                        geometric_min_inlier_ratio = 0.25,
                    ),
                ),
                global_pgo = SimpleNamespace(
                    lambda_switch  = 1.0,
                    max_iterations = 50,
                    huber_delta    = 0.5,
                    device         = "cpu",
                ),
                lc_min_gap              = 50,
                lc_geometric_threshold  = 0.70,
                keyframe_freq           = 5,
                apply_correction        = True,
            )

        # Loop-closure detector — dispatch by type string
        lc_cfg       = slam_cfg.loop_closure
        lc_type_name = getattr(lc_cfg, "type", "MixVPRLoopDetector")
        lc_cls       = _LOOP_DETECTOR_REGISTRY.get(lc_type_name)
        if lc_cls is None:
            raise ValueError(
                f"Unknown loop-closure detector '{lc_type_name}'. "
                f"Available: {list(_LOOP_DETECTOR_REGISTRY)}")
        loop_detector = lc_cls(lc_cfg.args)

        Logger.write("info",
            f"MACSLAM: using loop detector  {lc_type_name}")

        # Global PGO manager
        g = slam_cfg.global_pgo
        global_pgo = GlobalPGO_Optimizer(
            lambda_switch  = g.lambda_switch,
            max_iterations = g.max_iterations,
            huber_delta    = getattr(g, "huber_delta", 0.5),
            device         = g.device,
        )

        return cls(
            local_vo               = local_vo,
            loop_detector          = loop_detector,
            global_pgo             = global_pgo,
            lc_min_gap             = getattr(slam_cfg, "lc_min_gap", 50),
            lc_geometric_threshold = getattr(slam_cfg, "lc_geometric_threshold", 0.70),
            keyframe_freq          = getattr(slam_cfg, "keyframe_freq", 5),
            apply_correction       = getattr(slam_cfg, "apply_correction", True),
        )

    # ==================================================================
    # Hook:  local MAC-VO optimisation finished
    # ==================================================================

    def _on_local_optimize_done(self, system: MACVO) -> None:
        """
        Called *after* the local two-frame PGO writes its result to the map.

        Extracts the sequential edge  (T̃_{i,i+1} , Σ^{seq})  and pushes it
        to ``GlobalPGO_Optimizer``.

        TODO (plan §II, Optimizer.py edit):
            Replace the identity covariance below with the actual
            marginalized covariance  H⁻¹  from the local solver.
        """
        graph    = system.graph
        n_frames = len(graph.frames)
        if n_frames < 2:
            return

        curr_idx = n_frames - 1
        prev_idx = curr_idx - 1

        T_prev = pp.SE3(graph.frames.data["pose"][prev_idx].unsqueeze(0).double())
        T_curr = pp.SE3(graph.frames.data["pose"][curr_idx].unsqueeze(0).double())
        T_rel  = T_prev.Inv() @ T_curr       # T̃_{prev → curr}

        # Placeholder covariance  (see docstring above)
        cov = torch.eye(6, dtype=torch.float64) * 1e-4
        cov[3:, 3:] *= 0.1

        self.global_pgo.add_sequential_edge(SequentialEdge(
            kf_idx_from   = prev_idx,
            kf_idx_to     = curr_idx,
            relative_pose = T_rel,
            covariance    = cov,
        ))
        self.global_pgo.update_keyframe_pose(curr_idx, T_curr)
        if prev_idx == 0:
            self.global_pgo.update_keyframe_pose(0, T_prev)

    # ==================================================================
    # Loop-closure pipeline
    # ==================================================================

    def _check_loop_closure(self, kf_idx: int, frame: T_SensorFrame) -> None:
        """
        1. Encode the keyframe and add to VPR database.
        2. Query for candidates (cosine similarity).
        3. Geometrically verify each candidate (ORB + E-matrix RANSAC
           when ``MixVPRLoopDetector`` is used, heuristic fallback otherwise).
        4. Push verified LC edges → ``GlobalPGO_Optimizer``.
        5. Trigger a background solve if new LCs were found.
        """
        image = frame.stereo.imageL              # (1, C, H, W)

        # 1. Encode
        self.loop_detector.add_keyframe(kf_idx, image)

        # 2. Query
        candidates = self.loop_detector.query(kf_idx, min_gap=self.lc_min_gap)
        if not candidates:
            return

        Logger.write("info",
            f"MACSLAM: LC candidates for KF {kf_idx}: "
            + ", ".join(f"{c[0]}({c[1]:.2f})" for c in candidates))

        # 3 + 4.  Verify & push
        any_accepted = False
        for cand_kf_idx, similarity in candidates:
            try:
                lc = self._geometric_verify(
                    kf_idx, cand_kf_idx, frame, similarity)
                if lc is None:
                    self.total_lc_rejected += 1
                    continue
                self.global_pgo.add_loop_closure_edge(LoopClosureEdge(
                    kf_idx_a      = lc.query_kf_idx,
                    kf_idx_b      = lc.match_kf_idx,
                    relative_pose = lc.relative_pose,
                    covariance    = lc.covariance,
                    confidence    = lc.confidence,
                ))
                self.total_lc_detected += 1
                any_accepted = True
                Logger.write("info",
                    f"MACSLAM: LC accepted  {kf_idx} ↔ {cand_kf_idx}  "
                    f"(confidence={lc.confidence:.3f})")
            except Exception as e:
                Logger.write("warn",
                    f"MACSLAM: verification failed {kf_idx}→{cand_kf_idx}: {e}")

        # 5. Trigger global solve
        if any_accepted and not self.global_pgo.is_running:
            self.global_pgo.trigger_optimization()

    def _geometric_verify(
        self,
        query_kf_idx: int,
        match_kf_idx: int,
        query_frame:  T_SensorFrame,
        similarity:   float,
    ) -> LoopClosureResult | None:
        """
        Geometric verification of a loop-closure candidate.

        Primary path  (MixVPRLoopDetector)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Uses ORB keypoints + Essential Matrix RANSAC (via the detector's
        ``GeometricVerifier``) to check geometric consistency and recover
        the relative pose.  The covariance is scaled by the inlier ratio:
        more inliers → tighter covariance → stronger constraint in the
        global pose graph.

        Fallback path  (NetVLADLoopDetector / any detector without
        ``geometric_verify``)
        ~~~~~~~~~~~~~~~~~~~~~~
        Uses the poses already in the map to compute the relative transform
        and assigns a heuristic covariance that grows with the temporal gap.
        """
        graph    = self.local_vo.graph
        n_frames = len(graph.frames)
        if query_kf_idx >= n_frames or match_kf_idx >= n_frames:
            return None

        # ── Primary: ORB + Essential Matrix RANSAC ──────────────────────
        if hasattr(self.loop_detector, "geometric_verify"):
            K_tensor = graph.frames.data["K"][query_kf_idx]   # (3, 3)
            K_np = K_tensor.cpu().numpy().astype(np.float64)

            accepted, inlier_ratio, T_4x4 = self.loop_detector.geometric_verify(
                query_kf_idx, match_kf_idx, K_np)

            if not accepted or T_4x4 is None:
                Logger.write("debug",
                    f"MACSLAM: geometric verification rejected "
                    f"{query_kf_idx}→{match_kf_idx}  "
                    f"(inlier_ratio={inlier_ratio:.3f})")
                return None

            # Convert 4×4 numpy SE(3) → pypose SE3
            T_rel = pp.mat2SE3(
                torch.from_numpy(T_4x4).unsqueeze(0).double())

            # Covariance: tighter when more inliers agree
            #   inlier_ratio ≈ 1.0  →  cov_scale ≈ 0.001  (strong)
            #   inlier_ratio ≈ 0.25 →  cov_scale ≈ 0.038  (weak)
            cov_scale = max(0.001, 0.05 * (1.0 - inlier_ratio))
            cov = torch.eye(6, dtype=torch.float64) * cov_scale
            cov[3:, 3:] *= 0.1   # rotation block tighter than translation

            return LoopClosureResult(
                query_kf_idx  = query_kf_idx,
                match_kf_idx  = match_kf_idx,
                relative_pose = T_rel,
                covariance    = cov,
                confidence    = inlier_ratio,
            )

        # ── Fallback: pose-based heuristic ──────────────────────────────
        if similarity < self.lc_geometric_threshold:
            return None

        T_query = pp.SE3(
            graph.frames.data["pose"][query_kf_idx].unsqueeze(0).double())
        T_match = pp.SE3(
            graph.frames.data["pose"][match_kf_idx].unsqueeze(0).double())

        T_rel = T_match.Inv() @ T_query       # T̃_{match → query}

        # Heuristic covariance scaled by frame gap
        gap       = abs(query_kf_idx - match_kf_idx)
        scale     = min(gap / 100.0, 5.0)
        cov       = torch.zeros(6, 6, dtype=torch.float64)
        cov[:3, :3] = torch.eye(3) * 1e-2 * scale
        cov[3:, 3:] = torch.eye(3) * 1e-3 * scale

        return LoopClosureResult(
            query_kf_idx  = query_kf_idx,
            match_kf_idx  = match_kf_idx,
            relative_pose = T_rel,
            covariance    = cov,
            confidence    = similarity,
        )

    # ==================================================================
    # Correction propagation
    # ==================================================================

    def _apply_global_correction(self) -> None:
        """
        Poll the global PGO for a finished solve and propagate δT.

        For each keyframe  k:   T_k ← δT_k · T_k
        For non-keyframes between two corrected KFs: apply δT of the
        nearest *preceding* keyframe  (rigid correction).
        """
        if not self.apply_correction:
            return

        correction = self.global_pgo.get_correction()
        if correction is None:
            return

        graph    = self.local_vo.graph
        n_frames = len(graph.frames)

        Logger.write("info",
            f"MACSLAM: applying δT to {len(correction.kf_corrections)} KFs")

        # 1. Apply to keyframes
        for kf_idx, delta_T in correction.kf_corrections.items():
            if kf_idx >= n_frames:
                continue
            old = pp.SE3(graph.frames.data["pose"][kf_idx].unsqueeze(0).double())
            new = delta_T.double() @ old
            graph.frames.data["pose"][kf_idx] = \
                NormalizeQuat(new).float().squeeze(0)

        # 2. Propagate to non-keyframes
        sorted_kfs = sorted(correction.kf_corrections.keys())
        for i in range(n_frames):
            if i in correction.kf_corrections:
                continue
            # find nearest preceding corrected keyframe
            nearest = None
            for kf in sorted_kfs:
                if kf <= i:
                    nearest = kf
                else:
                    break
            if nearest is not None:
                dT  = correction.kf_corrections[nearest]
                old = pp.SE3(
                    graph.frames.data["pose"][i].unsqueeze(0).double())
                new = dT.double() @ old
                graph.frames.data["pose"][i] = \
                    NormalizeQuat(new).float().squeeze(0)

        Logger.write("info", "MACSLAM: corrections applied")

    # ==================================================================
    # IOdometry interface
    # ==================================================================

    @Timer.cpu_timeit("MACSLAM_Runtime")
    def run(self, frame: T_SensorFrame) -> None:
        """
        Main per-frame entry point.

        1. Run local MAC-VO  (Tier 1).
        2. On global-keyframe frames → encode for VPR + check LC.
        3. Poll for and apply global corrections if ready.
        """
        # Step 1
        self.local_vo.run(frame)
        self.frame_count += 1

        if not self.local_vo.isinitiated:
            return

        curr_kf_idx = len(self.local_vo.graph.frames) - 1

        # Step 2
        is_global_kf = (
            (self.frame_count % self.keyframe_freq == 0)
            or self.frame_count <= 1
        )
        if is_global_kf and curr_kf_idx >= 0:
            self.global_kf_indices.append(curr_kf_idx)
            self._check_loop_closure(curr_kf_idx, frame)

        # Step 3
        self._apply_global_correction()

    def get_map(self) -> VisualMap:
        return self.local_vo.get_map()

    def terminate(self) -> None:
        self.global_pgo.terminate()
        self._apply_global_correction()      # last chance for corrections
        self.local_vo.terminate()
        self.terminated = True
        Logger.write("info",
            f"MACSLAM: terminated — {self.total_lc_detected} loop closures "
            f"accepted, {self.total_lc_rejected} rejected, "
            f"across {self.frame_count} frames")