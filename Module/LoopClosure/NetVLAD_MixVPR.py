"""
Loop Closure Detectors for MAC-SLAM
====================================

Two detector implementations:

1. **MixVPRLoopDetector**  (recommended)
   Self-contained MixVPR (WACV 2023) re-implementation with pretrained
   GSV-Cities weights.  ResNet-50 backbone + Feature-Mixer aggregation ->
   4096-d descriptor.  94.6% R@1 on Pitts250k-test.

2. **NetVLADLoopDetector**  (lightweight fallback)
   GeM-pooled ResNet-18 -- untrained for VPR, only useful for quick tests.

Geometric Verification
~~~~~~~~~~~~~~~~~~~~~~
``GeometricVerifier`` estimates the essential matrix between two images
using ORB keypoints + brute-force matching + OpenCV RANSAC.  If the
inlier ratio exceeds a threshold, the relative pose is recovered from
the essential matrix and returned with a diagonal covariance scaled by
the inlier count.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Interface import ILoopClosureDetector, LoopClosureResult


# ===================================================================
#  MixVPR Architecture  (self-contained, no external repo needed)
# ===================================================================

class FeatureMixerLayer(nn.Module):
    """Single Feature-Mixer block from the MixVPR paper.

    Operates on (B, C, N) where C = channels, N = H*W spatial tokens.
    Mixes information *across spatial positions* via an MLP with skip.
    """

    def __init__(self, in_dim: int, mlp_ratio: int = 1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(x)


class MixVPRAggregator(nn.Module):
    """MixVPR aggregation head.

    Takes feature maps from an intermediate backbone layer, flattens the
    spatial dims, applies L stacked Feature-Mixer blocks, then projects
    to a compact descriptor.

    Args:
        in_channels:  Feature-map channels (1024 for ResNet50-layer3).
        in_h, in_w:   Spatial dims of the feature map (20x20 for 320 input).
        out_channels: Channels after row-wise projection.
        mix_depth:    Number of Feature-Mixer blocks (L=4 in paper).
        mlp_ratio:    MLP expansion ratio inside each mixer block.
        out_rows:     Number of output rows (desc_dim = out_rows * out_channels).
    """

    def __init__(
        self,
        in_channels: int = 1024, in_h: int = 20, in_w: int = 20,
        out_channels: int = 1024, mix_depth: int = 4,
        mlp_ratio: int = 1, out_rows: int = 4,
    ):
        super().__init__()
        hw = in_h * in_w
        self.mix_blocks = nn.ModuleList(
            [FeatureMixerLayer(hw, mlp_ratio) for _ in range(mix_depth)])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)                                 # (B, C, H*W)
        for mix in self.mix_blocks:
            x = mix(x)
        x = self.channel_proj(x.permute(0, 2, 1))        # (B, H*W, out_ch)
        x = x.permute(0, 2, 1)                           # (B, out_ch, H*W)
        x = self.row_proj(x)                              # (B, out_ch, out_rows)
        x = x.flatten(1)                                  # (B, desc_dim)
        return F.normalize(x, p=2, dim=-1)


class VPRModel(nn.Module):
    """Complete VPR model:  backbone + aggregator.

    Mirrors ``main.py`` from the official MixVPR repo so that pretrained
    checkpoints load directly via ``load_state_dict``.
    """

    def __init__(
        self,
        backbone_arch: str = "resnet50",
        pretrained: bool = True,
        layers_to_freeze: int = 2,
        layers_to_crop: list[int] | None = None,
        agg_arch: str = "MixVPR",
        agg_config: dict | None = None,
    ):
        super().__init__()
        if layers_to_crop is None:
            layers_to_crop = [4]
        if agg_config is None:
            agg_config = dict(
                in_channels=1024, in_h=20, in_w=20,
                out_channels=1024, mix_depth=4, mlp_ratio=1, out_rows=4)

        import torchvision.models as models
        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }
        builder = {"resnet18": models.resnet18,
                    "resnet34": models.resnet34,
                    "resnet50": models.resnet50}
        net = builder[backbone_arch](
            weights=weights_map[backbone_arch] if pretrained else None)

        layers = list(net.children())[:-2]
        # Crop: layer4 is child-index 7 -> crop_idx=4 means remove child[7]
        for ci in sorted(layers_to_crop or [], reverse=True):
            idx = ci + 3
            if idx < len(layers):
                layers = layers[:idx]
        self.backbone = nn.Sequential(*layers)

        for i, child in enumerate(self.backbone.children()):
            if i < layers_to_freeze + 3:
                for p in child.parameters():
                    p.requires_grad = False

        self.aggregator = MixVPRAggregator(**agg_config)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(x))


# ===================================================================
#  Geometric Verifier  (Essential Matrix + RANSAC)
# ===================================================================

class GeometricVerifier:
    """ORB features + Essential Matrix RANSAC for LC verification.

    Replaces the old "trust the VPR score" approach with actual
    geometric consistency checking.
    """

    def __init__(
        self,
        n_features: int = 1000,
        match_ratio: float = 0.75,
        ransac_threshold: float = 1.0,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.25,
    ):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

    @staticmethod
    def _to_gray(image: torch.Tensor) -> np.ndarray:
        img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if img.shape[2] == 3:
            return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return (img * 255).astype(np.uint8).squeeze(-1)

    def verify(
        self,
        img_q: torch.Tensor,
        img_m: torch.Tensor,
        K: np.ndarray,
    ) -> tuple[bool, float, Optional[np.ndarray]]:
        """
        Returns (accepted, inlier_ratio, T_4x4_or_None).
        """
        gq, gm = self._to_gray(img_q), self._to_gray(img_m)
        kp_q, des_q = self.orb.detectAndCompute(gq, None)
        kp_m, des_m = self.orb.detectAndCompute(gm, None)

        if des_q is None or des_m is None or len(kp_q) < 10 or len(kp_m) < 10:
            return False, 0.0, None

        raw = self.matcher.knnMatch(des_q, des_m, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2)
                if m.distance < self.match_ratio * n.distance]

        if len(good) < self.min_inliers:
            return False, len(good) / max(len(raw), 1), None

        pts_q = np.float32([kp_q[m.queryIdx].pt for m in good])
        pts_m = np.float32([kp_m[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts_q, pts_m, K, method=cv2.RANSAC,
            prob=0.999, threshold=self.ransac_threshold)
        if E is None or mask is None:
            return False, 0.0, None

        n_inliers = int(mask.sum())
        ratio = n_inliers / len(good)
        if n_inliers < self.min_inliers or ratio < self.min_inlier_ratio:
            return False, ratio, None

        _, R, t, _ = cv2.recoverPose(E, pts_q, pts_m, K, mask=mask)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return True, ratio, T


# ===================================================================
#  MixVPR Loop Detector  (recommended)
# ===================================================================

class MixVPRLoopDetector(ILoopClosureDetector):
    """
    Loop closure detector using pretrained MixVPR descriptors +
    geometric verification via ORB + Essential Matrix RANSAC.

    Config (required):
        device, threshold, top_k, min_gap
    Config (optional):
        weight_path, orb_features, match_ratio, ransac_threshold,
        geometric_min_inliers, geometric_min_inlier_ratio
    """

    DEFAULT_WEIGHT_NAME = "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.device = config.device
        self.threshold = getattr(config, "threshold", 0.70)
        self.top_k = getattr(config, "top_k", 3)
        self.default_min_gap = getattr(config, "min_gap", 50)

        # Build MixVPR model
        self.model = VPRModel(
            backbone_arch="resnet50", pretrained=True,
            layers_to_freeze=2, layers_to_crop=[4],
            agg_config=dict(
                in_channels=1024, in_h=20, in_w=20,
                out_channels=1024, mix_depth=4, mlp_ratio=1, out_rows=4))

        wp = getattr(config, "weight_path", None)
        if wp is None:
            wp = os.path.join("Model", self.DEFAULT_WEIGHT_NAME)
        self._load_weights(wp)
        self.model = self.model.to(self.device).eval()

        # Geometric verifier
        self.verifier = GeometricVerifier(
            n_features=getattr(config, "orb_features", 1000),
            match_ratio=getattr(config, "match_ratio", 0.75),
            ransac_threshold=getattr(config, "ransac_threshold", 1.0),
            min_inliers=getattr(config, "geometric_min_inliers", 30),
            min_inlier_ratio=getattr(config, "geometric_min_inlier_ratio", 0.25))

        # Database
        self.kf_indices: list[int] = []
        self.descriptors: list[torch.Tensor] = []
        self.kf_images: dict[int, torch.Tensor] = {}

        self._mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1).to(self.device)
        self._std  = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1).to(self.device)

    def _load_weights(self, path: str) -> None:
        from Utility.PrettyPrint import Logger
        p = Path(path)
        if p.exists():
            try:
                sd = torch.load(str(p), map_location="cpu", weights_only=False)
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                self.model.load_state_dict(sd, strict=False)
                Logger.write("info", f"MixVPR: loaded weights from {p}")
            except Exception as e:
                Logger.write("warn", f"MixVPR: failed to load {p}: {e}")
        else:
            Logger.write("warn",
                f"MixVPR: weights not found at {p}. "
                f"Download from the official MixVPR release: "
                f"https://github.com/amaralibey/MixVPR#weights "
                f"and place at {p}. Using ImageNet backbone only.")

    @torch.inference_mode()
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        x = image.to(self.device).float()
        x = (x - self._mean) / self._std
        x = F.interpolate(x, size=(320, 320), mode="bilinear", align_corners=False)
        return self.model(x).cpu()

    def add_keyframe(self, kf_idx: int, image: torch.Tensor) -> None:
        self.kf_indices.append(kf_idx)
        self.descriptors.append(self._encode(image))
        self.kf_images[kf_idx] = image.cpu().float()

    def query(self, kf_idx: int, min_gap: int | None = None) -> list[tuple[int, float]]:
        if min_gap is None:
            min_gap = self.default_min_gap
        if len(self.descriptors) < 2:
            return []
        try:
            qpos = self.kf_indices.index(kf_idx)
        except ValueError:
            return []
        qd = self.descriptors[qpos]
        sims = (torch.cat(self.descriptors, 0) @ qd.T).squeeze(-1)
        cands = [(ki, s) for ki, s in zip(self.kf_indices, sims.tolist())
                 if abs(ki - kf_idx) >= min_gap and s >= self.threshold]
        cands.sort(key=lambda x: -x[1])
        return cands[:self.top_k]

    def geometric_verify(
        self, query_kf_idx: int, match_kf_idx: int, K: np.ndarray,
    ) -> tuple[bool, float, Optional[np.ndarray]]:
        iq = self.kf_images.get(query_kf_idx)
        im = self.kf_images.get(match_kf_idx)
        if iq is None or im is None:
            return False, 0.0, None
        return self.verifier.verify(iq, im, K)

    def reset(self) -> None:
        self.kf_indices.clear()
        self.descriptors.clear()
        self.kf_images.clear()

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "device":    lambda s: isinstance(s, str),
            "threshold": lambda t: isinstance(t, float) and 0.0 < t < 1.0,
            "top_k":     lambda k: isinstance(k, int) and k > 0,
            "min_gap":   lambda g: isinstance(g, int) and g > 0,
        })


# ===================================================================
#  GeM-ResNet Fallback  (lightweight, low accuracy)
# ===================================================================

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)


class NetVLADLoopDetector(ILoopClosureDetector):
    """Lightweight fallback using GeM-pooled ResNet-18 (not VPR-trained)."""
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.device = config.device
        self.threshold = getattr(config, "threshold", 0.85)
        self.top_k = getattr(config, "top_k", 3)
        self.default_min_gap = getattr(config, "min_gap", 30)
        dim = getattr(config, "descriptor_dim", 512)
        import torchvision.models as models
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        enc = nn.Sequential(*list(net.children())[:-2])
        pool = GeM()
        with torch.no_grad():
            fd = pool(enc(torch.zeros(1,3,224,224))).shape[1]
        proj = nn.Linear(fd, dim) if fd != dim else nn.Identity()
        for p in enc.parameters(): p.requires_grad = False
        enc.eval()
        self.enc = enc.to(self.device)
        self.pool = pool.to(self.device)
        self.proj = proj.to(self.device)
        self.kf_indices: list[int] = []
        self.descriptors: list[torch.Tensor] = []
        self._mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(self.device)
        self._std  = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(self.device)

    @torch.inference_mode()
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        x = (image.to(self.device).float() - self._mean) / self._std
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        return F.normalize(self.proj(self.pool(self.enc(x))), p=2, dim=-1).cpu()

    def add_keyframe(self, kf_idx: int, image: torch.Tensor) -> None:
        self.kf_indices.append(kf_idx)
        self.descriptors.append(self._encode(image))

    def query(self, kf_idx: int, min_gap: int|None = None) -> list[tuple[int,float]]:
        if min_gap is None: min_gap = self.default_min_gap
        if len(self.descriptors) < 2: return []
        try: qpos = self.kf_indices.index(kf_idx)
        except ValueError: return []
        qd = self.descriptors[qpos]
        sims = (torch.cat(self.descriptors,0) @ qd.T).squeeze(-1)
        c = [(ki,s) for ki,s in zip(self.kf_indices,sims.tolist())
             if abs(ki-kf_idx)>=min_gap and s>=self.threshold]
        c.sort(key=lambda x: -x[1])
        return c[:self.top_k]

    def reset(self) -> None:
        self.kf_indices.clear(); self.descriptors.clear()

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace|None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "device": lambda s: isinstance(s,str),
            "backbone": lambda s: s in {"resnet18","resnet34","resnet50"},
            "descriptor_dim": lambda d: isinstance(d,int) and d>0,
            "threshold": lambda t: isinstance(t,float) and 0.<t<1.,
            "top_k": lambda k: isinstance(k,int) and k>0,
            "min_gap": lambda g: isinstance(g,int) and g>0,
        })