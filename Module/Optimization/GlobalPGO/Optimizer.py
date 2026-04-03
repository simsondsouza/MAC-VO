"""
Asynchronous Global Pose-Graph Optimizer  (Tier 2)
===================================================

Thread-safe manager that sits between the local MAC-VO (Tier 1) and the
``GlobalPoseGraph`` factor graph.

Lifecycle
---------
1.  **Edge collection** — the main thread pushes ``SequentialEdge``s
    (relative pose + Hessian-derived covariance from the local LM solver)
    and ``LoopClosureEdge``s (from geometric verification) via
    ``add_sequential_edge`` / ``add_loop_closure_edge``.

2.  **Trigger** — when a loop closure is detected the main thread calls
    ``trigger_optimization()``, which snapshots the current edges & poses
    and launches a background ``threading.Thread``.

3.  **Solve** — the background thread builds a ``GlobalPoseGraph`` and runs
    PyPose LM until convergence.

4.  **Correction** — the optimised keyframe poses T* are compared with the
    pre-solve estimates T̂ to produce corrections  δT = T* · T̂⁻¹ .
    The main thread polls ``get_correction()`` (non-blocking) and, when
    ready, applies the rigid correction to all keyframes and interpolates
    for non-keyframes.

Notes on the Hessian covariance
-------------------------------
The plan specifies extracting the *marginalized covariance* from the local
Hessian  H = J^T W J .  In the current MAC-VO ``TwoFrame_PGO`` the
Hessian ``A`` is already computed inside the LM step.  To expose it you
need to cache ``A_inv = torch.linalg.inv(A[:6,:6])`` at the end of each
local solve and attach it to the ``GraphOutput``.  Until that plumbing is
done a scaled-identity fallback is used (see ``_default_seq_cov``).
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion
from pypose.optim.solver import Cholesky

from Utility.PrettyPrint import Logger
from Utility.Math import NormalizeQuat

from .Graphs import GlobalPoseGraph, GlobalGraphInput, GlobalGraphOutput


# ═══════════════════════════════════════════════════════════════════════════
# Data containers exchanged between Tier-1 (local) and Tier-2 (global)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SequentialEdge:
    """Sequential (odometry) edge between adjacent keyframes."""
    kf_idx_from:    int              # global kf index  k_i
    kf_idx_to:      int              # global kf index  k_{i+1}
    relative_pose:  pp.LieTensor     # T̃_{i,i+1}   (1, 7) SE3
    covariance:     torch.Tensor     # Σ^{seq}      (6, 6) float64


@dataclass
class LoopClosureEdge:
    """Loop-closure edge between two distant keyframes."""
    kf_idx_a:       int
    kf_idx_b:       int
    relative_pose:  pp.LieTensor     # T̃_{a,b}     (1, 7) SE3
    covariance:     torch.Tensor     # Σ^{LC}       (6, 6) float64
    confidence:     float            # VPR similarity / inlier ratio


@dataclass
class CorrectionResult:
    """Rigid corrections produced after global optimisation converges."""
    kf_corrections: dict[int, pp.LieTensor]           # kf_idx → δT
    switch_values:  dict[tuple[int, int], float]       # (a,b) → final s_ab


# ═══════════════════════════════════════════════════════════════════════════
# Default covariance when the Hessian inverse is not yet wired through
# ═══════════════════════════════════════════════════════════════════════════

def _default_seq_cov(scale: float = 1e-4) -> torch.Tensor:
    """Isotropic 6×6 covariance (translation + rotation)."""
    cov = torch.eye(6, dtype=torch.float64) * scale
    cov[3:, 3:] *= 0.1          # rotation is typically tighter
    return cov


# ═══════════════════════════════════════════════════════════════════════════
# GlobalPGO_Optimizer  (a.k.a.  GlobalPGOManager in the plan)
# ═══════════════════════════════════════════════════════════════════════════

class GlobalPGO_Optimizer:
    """
    Thread-safe orchestrator for the global pose-graph back-end.

    Public API (called from the **main** thread)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``update_keyframe_pose(kf_idx, pose)``
        Keep the manager informed of the latest local estimate.

    ``add_sequential_edge(edge)``
        Feed a new odometry factor.

    ``add_loop_closure_edge(edge)``
        Feed a geometrically-verified loop-closure factor.

    ``trigger_optimization()``
        Snapshot edges and launch a background solve (non-blocking).

    ``get_correction() → CorrectionResult | None``
        Poll for results.  Returns ``None`` while the solver is still
        running or if no results are available.

    ``terminate()``
        Block until any running solve finishes (called on shutdown).

    Parameters
    ----------
    lambda_switch : float
        Penalty weight λ_s for the switchable-constraint prior.
        Higher ⇒ harder to disable a loop closure.
    max_iterations : int
        Upper bound on LM iterations per solve.
    huber_delta : float
        Huber kernel threshold applied inside LM.
    device : str
        ``'cpu'`` is recommended — the global graph is sparse and
        CPU solvers are fine for ≤1 k keyframes.
    """

    def __init__(
        self,
        lambda_switch:  float = 1.0,
        max_iterations: int   = 50,
        huber_delta:    float = 0.5,
        device:         str   = "cpu",
    ) -> None:
        self.lambda_switch  = lambda_switch
        self.max_iterations = max_iterations
        self.huber_delta    = huber_delta
        self.device         = device

        # ── Thread-safe shared state ────────────────────────────────────
        self._lock       = threading.Lock()
        self.seq_edges:  list[SequentialEdge]   = []
        self.lc_edges:   list[LoopClosureEdge]  = []
        self.kf_poses:   dict[int, pp.LieTensor] = {}   # kf_idx → pose

        # ── Background thread bookkeeping ───────────────────────────────
        self._opt_thread:  Optional[threading.Thread] = None
        self._correction:  Optional[CorrectionResult] = None
        self._is_running:  bool = False

    # ------------------------------------------------------------------
    # Pose tracking
    # ------------------------------------------------------------------

    def update_keyframe_pose(self, kf_idx: int, pose: pp.LieTensor) -> None:
        """Store / refresh the latest local pose estimate for *kf_idx*."""
        with self._lock:
            self.kf_poses[kf_idx] = pp.SE3(
                pose.tensor().clone().detach().double())

    # ------------------------------------------------------------------
    # Edge insertion
    # ------------------------------------------------------------------

    def add_sequential_edge(self, edge: SequentialEdge) -> None:
        """Append a sequential factor  (main thread)."""
        with self._lock:
            self.seq_edges.append(edge)
        # Logger.write("info",
        #     f"GlobalPGO: seq edge  {edge.kf_idx_from} → {edge.kf_idx_to}")

    def add_loop_closure_edge(self, edge: LoopClosureEdge) -> None:
        """Append a loop-closure factor  (main thread)."""
        with self._lock:
            self.lc_edges.append(edge)
        Logger.write("info",
            f"GlobalPGO: LC  edge  {edge.kf_idx_a} → {edge.kf_idx_b}  "
            f"(conf={edge.confidence:.3f})")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the background solver is active."""
        return self._is_running

    @property
    def has_loop_closures(self) -> bool:
        with self._lock:
            return len(self.lc_edges) > 0

    @property
    def num_sequential(self) -> int:
        with self._lock:
            return len(self.seq_edges)

    @property
    def num_loop_closures(self) -> int:
        with self._lock:
            return len(self.lc_edges)

    # ------------------------------------------------------------------
    # Trigger / poll
    # ------------------------------------------------------------------

    def trigger_optimization(self) -> None:
        """Snapshot current state and launch a background solve."""
        if self._is_running:
            Logger.write("warn",
                "GlobalPGO: solve already running — skipping trigger")
            return

        self._is_running = True
        self._correction = None
        self._opt_thread = threading.Thread(
            target=self._run_optimization, daemon=True)
        self._opt_thread.start()
        Logger.write("info", "GlobalPGO: background solve triggered")

    def get_correction(self, block: bool = False) -> Optional[CorrectionResult]:
        """Returns the correction. If block=True, waits for the solver to finish."""
        # Block the main thread if requested and the solver is running
        if block and self._is_running and self._opt_thread is not None:
            Logger.write("info", "GlobalPGO: Blocking main thread to wait for global solve...")
            self._opt_thread.join()

        if self._is_running:
            return None
        
        result = self._correction
        self._correction = None
        return result

    # ------------------------------------------------------------------
    # Background solver  (runs on self._opt_thread)
    # ------------------------------------------------------------------

    def _run_optimization(self) -> None:
        try:
            self._solve_global_graph()
        except Exception:
            Logger.write("error",
                f"GlobalPGO: optimisation failed\n{traceback.format_exc()}")
        finally:
            self._is_running = False

    def _solve_global_graph(self) -> None:
        # ── 1.  Snapshot under lock ─────────────────────────────────────
        with self._lock:
            poses_snap = dict(self.kf_poses)
            seq_snap   = list(self.seq_edges)
            lc_snap    = list(self.lc_edges)

        if len(seq_snap) == 0:
            Logger.write("warn", "GlobalPGO: no sequential edges — abort")
            return

        # ── 2.  Collect unique keyframe indices ─────────────────────────
        kf_set: set[int] = set()
        for e in seq_snap:
            kf_set.update((e.kf_idx_from, e.kf_idx_to))
        for e in lc_snap:
            kf_set.update((e.kf_idx_a, e.kf_idx_b))

        kf_list    = sorted(kf_set)
        kf2local   = {kf: i for i, kf in enumerate(kf_list)}
        n_kf       = len(kf_list)

        # ── 3.  Build initial pose tensor ───────────────────────────────
        init_poses = torch.zeros(n_kf, 7, dtype=torch.float64)
        for kf_idx in kf_list:
            loc = kf2local[kf_idx]
            if kf_idx in poses_snap:
                init_poses[loc] = poses_snap[kf_idx].tensor().double().squeeze(0)
            else:
                init_poses[loc] = pp.identity_SE3(1).tensor().double().squeeze(0)

        # ── 4.  Translate edges into local-index space ──────────────────
        seq_pairs:     list[tuple[int, int]] = []
        seq_rel_poses: list[pp.LieTensor]    = []
        seq_covs:      list[torch.Tensor]    = []

        for e in seq_snap:
            if e.kf_idx_from in kf2local and e.kf_idx_to in kf2local:
                seq_pairs.append(
                    (kf2local[e.kf_idx_from], kf2local[e.kf_idx_to]))
                seq_rel_poses.append(
                    pp.SE3(e.relative_pose.tensor().double()))
                seq_covs.append(e.covariance.double())

        lc_pairs:     list[tuple[int, int]] = []
        lc_rel_poses: list[pp.LieTensor]    = []
        lc_covs:      list[torch.Tensor]    = []

        for e in lc_snap:
            if e.kf_idx_a in kf2local and e.kf_idx_b in kf2local:
                lc_pairs.append(
                    (kf2local[e.kf_idx_a], kf2local[e.kf_idx_b]))
                lc_rel_poses.append(
                    pp.SE3(e.relative_pose.tensor().double()))
                lc_covs.append(e.covariance.double())

        # ── 5.  Assemble GlobalGraphInput ───────────────────────────────
        graph_input = GlobalGraphInput(
            kf_poses      = pp.SE3(init_poses),
            kf_indices    = kf_list,
            seq_edges     = seq_pairs,
            seq_rel_poses = seq_rel_poses,
            seq_covs      = seq_covs,
            lc_edges      = lc_pairs,
            lc_rel_poses  = lc_rel_poses,
            lc_covs       = lc_covs,
            lambda_switch = self.lambda_switch,
            fix_first     = True,
        )

        # ── 6.  Build factor graph & LM optimiser ──────────────────────
        graph = GlobalPoseGraph(graph_input).to(dtype=torch.float64)

        

        optimizer = LM(
            graph,
            solver    = Cholesky(),
            strategy  = TrustRegion(radius=1e3),
            kernel    = Huber(delta=self.huber_delta),
            corrector = FastTriggs(Huber(delta=self.huber_delta)),
            min       = 1e-6,
            vectorize = True, 
        )
        scheduler = StopOnPlateau(
            optimizer,
            steps      = self.max_iterations,
            patience   = 3,
            decreasing = 1e-5,
            verbose    = False,
        )

        Logger.write("info",
            f"GlobalPGO: solving  {n_kf} KFs, "
            f"{len(seq_pairs)} seq, {len(lc_pairs)} LC")

        # ── 7.  Iterate ────────────────────────────────────────────────
        while scheduler.continual():
            loss = optimizer.step(input=())
            scheduler.step(loss)

        # ── 8.  Extract optimised poses & switch values ─────────────────
        result: GlobalGraphOutput = graph.write_back()

        # ── 9.  Compute corrections  δT = T* · T̂⁻¹ ────────────────────
        corrections: dict[int, pp.LieTensor] = {}
        for i, kf_idx in enumerate(kf_list):
            if kf_idx not in poses_snap:
                continue
            T_star = pp.SE3(result.optimized_poses[i : i + 1].tensor())
            T_hat  = poses_snap[kf_idx]
            delta  = T_star @ T_hat.Inv()
            corrections[kf_idx] = NormalizeQuat(delta)

        switch_dict: dict[tuple[int, int], float] = {}
        for k, e in enumerate(lc_snap):
            if k < result.switch_values.numel():
                switch_dict[(e.kf_idx_a, e.kf_idx_b)] = \
                    result.switch_values[k].item()

        self._correction = CorrectionResult(
            kf_corrections = corrections,
            switch_values  = switch_dict,
        )

        # ── 10. Log summary ────────────────────────────────────────────
        Logger.write("info",
            f"GlobalPGO: converged — corrections for "
            f"{len(corrections)} keyframes")
        for (a, b), s in switch_dict.items():
            tag = "ACCEPT" if s > 0.5 else "REJECT"
            Logger.write("info", f"  LC ({a}→{b}): s={s:.3f} [{tag}]")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def terminate(self) -> None:
        """Block until any running solve finishes (called at system exit)."""
        if self._opt_thread is not None and self._opt_thread.is_alive():
            Logger.write("info",
                "GlobalPGO: waiting for background solve to finish …")
            self._opt_thread.join(timeout=60.0)
            if self._opt_thread.is_alive():
                Logger.write("warn",
                    "GlobalPGO: background thread did not finish in time")