"""
Global Pose Graph with switchable constraints for loop closure.

The global cost function (Eq. from plan):

    L_global = Σ_i r_seq^T Σ_seq^{-1} r_seq                         (sequential edges)
             + Σ_(a,b) s²_ab r_LC^T Σ_LC^{-1} r_LC + λ_s(1-s_ab)²   (loop closure edges)

where:
    r_seq_{i,i+1} = Log( T̃_{i,i+1}^{-1} · T_{k_i}^{-1} · T_{k_{i+1}} )  ∈ ℝ⁶
    r_LC_{a,b}    = Log( T̃_{a,b}^{-1}   · T_a^{-1}     · T_b )          ∈ ℝ⁶
    s_ab ∈ [0,1]  is the switchable constraint variable
"""

import torch
import torch.nn as nn
import pypose as pp
from dataclasses import dataclass

from Module.Optimization.PyposeOptimizers import FactorGraph


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GlobalGraphInput:
    """All data required to build and solve the global pose graph."""

    kf_poses:       pp.LieTensor           # (N, 7) SE3 initial pose estimates
    kf_indices:     list[int]              # position-i  →  global kf_idx

    # Sequential edges produced by Local MAC-VO
    seq_edges:      list[tuple[int, int]]  # (local_from, local_to) index pairs
    seq_rel_poses:  list[pp.LieTensor]     # T̃_{i,i+1}  each (1,7) SE3
    seq_covs:       list[torch.Tensor]     # Σ^{seq}    each (6,6) float64

    # Loop-closure edges
    lc_edges:       list[tuple[int, int]]  # (local_a, local_b) index pairs
    lc_rel_poses:   list[pp.LieTensor]     # T̃_{a,b}    each (1,7) SE3
    lc_covs:        list[torch.Tensor]     # Σ^{LC}     each (6,6) float64

    lambda_switch:  float = 1.0            # λ_s  penalty for disabling LC
    fix_first:      bool  = True           # anchor the first keyframe


@dataclass
class GlobalGraphOutput:
    """Results after the global optimiser converges."""

    optimized_poses: pp.LieTensor          # (N, 7) corrected SE3 poses
    kf_indices:      list[int]             # same mapping as input
    switch_values:   torch.Tensor          # (n_lc,) final s_ab values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_cholesky_weight(info: torch.Tensor) -> torch.Tensor:
    """Return L^T from Cholesky of the information matrix, with a diagonal
    fallback when the matrix is not positive-definite."""
    try:
        L = torch.linalg.cholesky(info)
        return L.mT                                     # L^T  (6,6)
    except torch.linalg.LinAlgError:
        return torch.diag(torch.sqrt(torch.diag(info).clamp(min=1e-8)))


# ---------------------------------------------------------------------------
# Factor graph
# ---------------------------------------------------------------------------

class GlobalPoseGraph(FactorGraph):
    """
    PyTorch ``nn.Module`` that computes the full residual vector for the
    Levenberg–Marquardt solver.

    Parameters to optimise
    ~~~~~~~~~~~~~~~~~~~~~~
    * ``self.poses``        — ``pp.Parameter``  (N, 7)   SE3 keyframe poses
    * ``self.switch_vars``  — ``nn.Parameter``  (n_lc,)  switchable variables

    Forward output
    ~~~~~~~~~~~~~~
    A single 1-D tensor of stacked, *information-weighted* residuals:

        [ w·r_seq₀ ,  w·r_seq₁ ,  … ,
          s·w·r_lc₀,  √λ(1-s₀) ,
          s·w·r_lc₁,  √λ(1-s₁) ,  … ,
          anchor_residual ]

    Because the weighting (Cholesky of Σ⁻¹) is already baked in,
    ``covariance_array()`` returns identity so LM does not double-weight.
    """

    def __init__(self, data: GlobalGraphInput) -> None:
        super().__init__()
        self.data       = data
        self.n_kf       = data.kf_poses.shape[0]
        self.n_seq      = len(data.seq_edges)
        self.n_lc       = len(data.lc_edges)
        self.fix_first  = data.fix_first
        self.sqrt_lam   = (data.lambda_switch) ** 0.5

        # ── Optimisable parameters ──────────────────────────────────────
        self.poses = pp.Parameter(pp.SE3(data.kf_poses.clone().detach()))

        if self.n_lc > 0:
            self.switch_vars = nn.Parameter(
                torch.ones(self.n_lc, dtype=torch.float64))
        else:
            self.switch_vars = nn.Parameter(
                torch.empty(0, dtype=torch.float64))

        # ── Pre-compute & register buffers for edge data ────────────────
        # Sequential
        for k, (rel, cov) in enumerate(
                zip(data.seq_rel_poses, data.seq_covs)):
            self.register_buffer(
                f"seq_rel_inv_{k}",
                pp.SE3(rel.tensor().detach().clone().double()).Inv().tensor())
            self.register_buffer(
                f"seq_weight_{k}",
                _safe_cholesky_weight(torch.linalg.inv(cov.double())))

        # Loop closure
        for k, (rel, cov) in enumerate(
                zip(data.lc_rel_poses, data.lc_covs)):
            self.register_buffer(
                f"lc_rel_inv_{k}",
                pp.SE3(rel.tensor().detach().clone().double()).Inv().tensor())
            self.register_buffer(
                f"lc_weight_{k}",
                _safe_cholesky_weight(torch.linalg.inv(cov.double())))

        # Anchor (copy of the initial first pose)
        if self.fix_first:
            self.register_buffer(
                "anchor_pose",
                data.kf_poses[0:1].clone().detach().double())

    # ------------------------------------------------------------------
    # Core residual computation  (matches Eq. in the plan)
    # ------------------------------------------------------------------

    @staticmethod
    def _se3_residual(
        T_meas_inv: pp.LieTensor,
        T_from:     pp.LieTensor,
        T_to:       pp.LieTensor,
    ) -> torch.Tensor:
        """r = Log( T̃⁻¹ · T_from⁻¹ · T_to )  ∈ ℝ⁶"""
        return (T_meas_inv @ T_from.Inv() @ T_to).Log().tensor().squeeze(0)

    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:                          # noqa: D102
        parts: list[torch.Tensor] = []
        poses = self.poses                                      # (N, 7)

        # ── Sequential edges ────────────────────────────────────────────
        for k, (i_from, i_to) in enumerate(self.data.seq_edges):
            T_meas_inv = pp.SE3(getattr(self, f"seq_rel_inv_{k}"))
            W          = getattr(self, f"seq_weight_{k}")       # (6,6)

            r = self._se3_residual(
                    T_meas_inv,
                    pp.SE3(poses[i_from].unsqueeze(0)),
                    pp.SE3(poses[i_to  ].unsqueeze(0)))         # (6,)
            parts.append(W @ r)                                 # weighted

        # ── Loop-closure edges (with switchable constraints) ────────────
        for k, (i_a, i_b) in enumerate(self.data.lc_edges):
            T_meas_inv = pp.SE3(getattr(self, f"lc_rel_inv_{k}"))
            W          = getattr(self, f"lc_weight_{k}")        # (6,6)

            r = self._se3_residual(
                    T_meas_inv,
                    pp.SE3(poses[i_a].unsqueeze(0)),
                    pp.SE3(poses[i_b].unsqueeze(0)))            # (6,)

            s = self.switch_vars[k].clamp(0.0, 1.0)
            parts.append(s * (W @ r))                           # s² via s·(W·r)

            # Switch penalty:  √λ_s · (1 - s_ab)
            parts.append((self.sqrt_lam * (1.0 - s)).unsqueeze(0))

        # ── First-pose anchor (strong prior) ────────────────────────────
        if self.fix_first:
            init_first = pp.SE3(self.anchor_pose)
            curr_first = pp.SE3(poses[0].unsqueeze(0))
            anchor_r   = (init_first.Inv() @ curr_first).Log().tensor().squeeze(0)
            parts.append(anchor_r * 1e2)                        # large weight

        # ── Assemble ────────────────────────────────────────────────────
        if len(parts) == 0:
            return torch.zeros(1, dtype=torch.float64)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Interface required by FactorGraph / LM
    # ------------------------------------------------------------------

    @torch.no_grad()
    def covariance_array(self) -> torch.Tensor:
        """Identity — real weighting is baked into the residuals."""
        n = self.forward().numel()
        return torch.eye(n, dtype=torch.float64).unsqueeze(0).expand(n, -1, -1)

    @torch.no_grad()
    def write_back(self) -> GlobalGraphOutput:
        return GlobalGraphOutput(
            optimized_poses=pp.SE3(self.poses.data.clone()),
            kf_indices=self.data.kf_indices,
            switch_values=(self.switch_vars.data.clone().clamp(0.0, 1.0)
                           if self.n_lc > 0
                           else torch.empty(0)),
        )