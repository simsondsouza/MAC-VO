"""
Global Pose Graph with switchable constraints for loop closure.
===============================================================

**Vectorized** implementation — all sequential and loop-closure edges are
batched into tensors so that ``forward()`` runs in O(1) Python calls
(pure tensor ops), not O(N_edges) Python for-loops.

Cost function (Eq. from plan):

    L = Σ_i  r_seq^T Σ_seq^{-1} r_seq
      + Σ_(a,b)  s²  r_LC^T Σ_LC^{-1} r_LC  +  λ(1-s)²

where:
    r_seq_{i,i+1} = Log( T̃_{i,i+1}^{-1} · T_{k_i}^{-1} · T_{k_{i+1}} )
    r_LC_{a,b}    = Log( T̃_{a,b}^{-1}   · T_a^{-1}     · T_b )
"""

import torch
import torch.nn as nn
import pypose as pp
from dataclasses import dataclass

from Module.Optimization.PyposeOptimizers import FactorGraph


# ───────────────────────────────────────────────────────────────────────
# Data containers
# ───────────────────────────────────────────────────────────────────────

@dataclass
class GlobalGraphInput:
    kf_poses:       pp.LieTensor           # (N, 7)
    kf_indices:     list[int]
    seq_edges:      list[tuple[int, int]]
    seq_rel_poses:  list[pp.LieTensor]     # each (1,7)
    seq_covs:       list[torch.Tensor]     # each (6,6)
    lc_edges:       list[tuple[int, int]]
    lc_rel_poses:   list[pp.LieTensor]
    lc_covs:        list[torch.Tensor]
    lambda_switch:  float = 1.0
    fix_first:      bool  = True


@dataclass
class GlobalGraphOutput:
    optimized_poses: pp.LieTensor
    kf_indices:      list[int]
    switch_values:   torch.Tensor


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _batch_cholesky_weights(covs: list[torch.Tensor]) -> torch.Tensor:
    """Stack covariance list → (E,6,6) and return batch Cholesky L^T of
    the information matrices.  Falls back to diagonal sqrt per-edge."""
    if len(covs) == 0:
        return torch.empty(0, 6, 6, dtype=torch.float64)
    stacked = torch.stack(covs).double()                    # (E, 6, 6)
    info = torch.linalg.inv(stacked)                        # (E, 6, 6)
    try:
        L = torch.linalg.cholesky(info)                     # (E, 6, 6)
        return L.mT                                         # L^T
    except torch.linalg.LinAlgError:
        # per-edge fallback
        weights = torch.zeros_like(info)
        for i in range(info.shape[0]):
            try:
                weights[i] = torch.linalg.cholesky(info[i]).mT
            except torch.linalg.LinAlgError:
                weights[i] = torch.diag(
                    torch.sqrt(torch.diag(info[i]).clamp(min=1e-8)))
        return weights


def _batch_se3_residual(
    T_meas_inv: pp.LieTensor,          # (E, 7)
    T_from:     pp.LieTensor,           # (E, 7)
    T_to:       pp.LieTensor,           # (E, 7)
) -> torch.Tensor:
    """Batched r = Log( T̃⁻¹ · T_from⁻¹ · T_to )  → (E, 6)"""
    return (T_meas_inv @ T_from.Inv() @ T_to).Log().tensor()   # (E, 6)


# ───────────────────────────────────────────────────────────────────────
# Vectorized Factor Graph
# ───────────────────────────────────────────────────────────────────────

class GlobalPoseGraph(FactorGraph):
    """
    Vectorized global pose graph.  All edges are stored as batched
    tensors; ``forward()`` is a handful of batched SE3 / matmul ops —
    no Python for-loops over edges.
    """

    def __init__(self, data: GlobalGraphInput) -> None:
        super().__init__()
        self.data      = data
        self.n_kf      = data.kf_poses.shape[0]
        self.n_seq     = len(data.seq_edges)
        self.n_lc      = len(data.lc_edges)
        self.fix_first = data.fix_first
        self.sqrt_lam  = data.lambda_switch ** 0.5

        # ── Optimisable parameters ──────────────────────────────────────
        self.poses = pp.Parameter(pp.SE3(data.kf_poses.clone().detach()))

        if self.n_lc > 0:
            self.switch_vars = nn.Parameter(
                torch.ones(self.n_lc, dtype=torch.float64))
        else:
            self.switch_vars = nn.Parameter(
                torch.empty(0, dtype=torch.float64))

        # ── Batch sequential edge data ──────────────────────────────────
        if self.n_seq > 0:
            seq_from = torch.tensor([e[0] for e in data.seq_edges], dtype=torch.long)
            seq_to   = torch.tensor([e[1] for e in data.seq_edges], dtype=torch.long)
            seq_meas_inv = pp.SE3(torch.cat(
                [pp.SE3(r.tensor().double()).Inv().tensor()
                 for r in data.seq_rel_poses], dim=0))          # (E_seq, 7)
            seq_W = _batch_cholesky_weights(data.seq_covs)      # (E_seq, 6, 6)

            self.register_buffer("seq_from_idx", seq_from)
            self.register_buffer("seq_to_idx",   seq_to)
            self.register_buffer("seq_meas_inv", seq_meas_inv.tensor())
            self.register_buffer("seq_W",        seq_W)

        # ── Batch loop-closure edge data ────────────────────────────────
        if self.n_lc > 0:
            lc_a = torch.tensor([e[0] for e in data.lc_edges], dtype=torch.long)
            lc_b = torch.tensor([e[1] for e in data.lc_edges], dtype=torch.long)
            lc_meas_inv = pp.SE3(torch.cat(
                [pp.SE3(r.tensor().double()).Inv().tensor()
                 for r in data.lc_rel_poses], dim=0))
            lc_W = _batch_cholesky_weights(data.lc_covs)

            self.register_buffer("lc_a_idx",    lc_a)
            self.register_buffer("lc_b_idx",    lc_b)
            self.register_buffer("lc_meas_inv", lc_meas_inv.tensor())
            self.register_buffer("lc_W",        lc_W)

        # ── Anchor ──────────────────────────────────────────────────────
        if self.fix_first:
            self.register_buffer(
                "anchor_pose",
                data.kf_poses[0:1].clone().detach().double())

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        poses = self.poses                                  # (N, 7)

        # ── Sequential edges  (batched) ─────────────────────────────────
        if self.n_seq > 0:
            T_from = pp.SE3(poses[self.seq_from_idx])       # (E, 7)
            T_to   = pp.SE3(poses[self.seq_to_idx])         # (E, 7)
            T_minv = pp.SE3(self.seq_meas_inv)              # (E, 7)

            r_seq = _batch_se3_residual(T_minv, T_from, T_to)  # (E, 6)
            # Weighted: W @ r  for each edge  →  (E, 6)
            r_seq_w = torch.bmm(
                self.seq_W, r_seq.unsqueeze(-1)).squeeze(-1)
            parts.append(r_seq_w.flatten())                 # (E*6,)

        # ── Loop-closure edges  (batched, switchable) ───────────────────
        if self.n_lc > 0:
            T_a    = pp.SE3(poses[self.lc_a_idx])
            T_b    = pp.SE3(poses[self.lc_b_idx])
            T_minv = pp.SE3(self.lc_meas_inv)

            r_lc = _batch_se3_residual(T_minv, T_a, T_b)   # (E_lc, 6)
            r_lc_w = torch.bmm(
                self.lc_W, r_lc.unsqueeze(-1)).squeeze(-1)  # (E_lc, 6)

            s = torch.sigmoid(self.switch_vars)            # (E_lc,)
            r_lc_sw = s.unsqueeze(-1) * r_lc_w              # (E_lc, 6)
            parts.append(r_lc_sw.flatten())

            # Switch penalties:  √λ · (1 - s)
            penalties = self.sqrt_lam * (1.0 - s)           # (E_lc,)
            parts.append(penalties)

        # ── First-pose anchor ───────────────────────────────────────────
        if self.fix_first:
            init_first = pp.SE3(self.anchor_pose)
            curr_first = pp.SE3(poses[0].unsqueeze(0))
            anchor_r = (init_first.Inv() @ curr_first).Log().tensor().squeeze(0)
            parts.append(anchor_r * 1e2)

        if len(parts) == 0:
            return torch.zeros(1, dtype=torch.float64)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def covariance_array(self) -> torch.Tensor:
        n = self.forward().numel()
        return torch.eye(n, dtype=torch.float64).unsqueeze(0)

    @torch.no_grad()
    def write_back(self) -> GlobalGraphOutput:
        return GlobalGraphOutput(
            optimized_poses=pp.SE3(self.poses.data.clone()),
            kf_indices=self.data.kf_indices,
            switch_values=(torch.sigmoid(self.switch_vars.data.clone())
                           if self.n_lc > 0
                           else torch.empty(0)),
        )