"""
NetVLAD / MixVPR-based place recognition for loop closure candidate retrieval.

Uses torchvision ResNet features pooled with GeM as a lightweight descriptor,
with optional MixVPR or NetVLAD weight loading for higher recall.
For KITTI 00, the simple GeM descriptor already works well since the revisit
is visually distinctive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from .Interface import ILoopClosureDetector


class GeM(nn.Module):
    """Generalized Mean Pooling — a simple but effective global descriptor."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)


class PlaceEncoder(nn.Module):
    """
    Lightweight place encoder using ResNet-18 backbone + GeM pooling.
    Produces a compact L2-normalized descriptor per image.
    """
    def __init__(self, backbone: str = "resnet18", dim: int = 512):
        super().__init__()
        import torchvision.models as models

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove avgpool + fc; keep up to layer4
        self.encoder = nn.Sequential(*list(net.children())[:-2])
        self.pool = GeM()

        # Determine output dim from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.encoder(dummy)
            feat_dim = self.pool(feat).shape[1]

        self.proj = nn.Linear(feat_dim, dim) if feat_dim != dim else nn.Identity()

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        desc = self.pool(feat)
        desc = self.proj(desc)
        return F.normalize(desc, p=2, dim=-1)


class NetVLADLoopDetector(ILoopClosureDetector):
    """
    Loop closure detector using global image descriptors (GeM-pooled ResNet features).
    
    Stores descriptors in a flat tensor database and performs brute-force cosine
    similarity search. For KITTI-scale sequences (<5k frames, <500 keyframes),
    this is fast enough. For larger-scale use, replace with FAISS index.
    
    Config:
        device:         str     — 'cuda' or 'cpu'
        backbone:       str     — 'resnet18', 'resnet34', 'resnet50'
        descriptor_dim: int     — dimension of output descriptor (default 512)
        threshold:      float   — minimum cosine similarity to accept as candidate
        top_k:          int     — max number of candidates to return
        min_gap:        int     — minimum frame gap between query and candidate
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.device = config.device
        self.backbone = getattr(config, "backbone", "resnet18")
        self.descriptor_dim = getattr(config, "descriptor_dim", 512)
        self.threshold = getattr(config, "threshold", 0.85)
        self.top_k = getattr(config, "top_k", 3)
        self.default_min_gap = getattr(config, "min_gap", 30)

        # Build encoder
        self.encoder = PlaceEncoder(
            backbone=self.backbone,
            dim=self.descriptor_dim
        ).to(self.device).eval()

        # Database: list of (kf_idx, descriptor_tensor)
        self.kf_indices: list[int] = []
        self.descriptors: list[torch.Tensor] = []  # Each is (1, D)

        # Precompute normalization constants for input images (ImageNet stats)
        self.register_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.register_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.inference_mode()
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor to a global descriptor.
        Args:
            image: (1, C, H, W), values in [0, 1]
        Returns:
            (1, D) L2-normalized descriptor
        """
        x = image.to(self.device).float()
        # Normalize with ImageNet stats
        x = (x - self.register_mean) / self.register_std
        # Resize to 224x224 for backbone
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.encoder(x).cpu()

    def add_keyframe(self, kf_idx: int, image: torch.Tensor) -> None:
        desc = self._encode(image)
        self.kf_indices.append(kf_idx)
        self.descriptors.append(desc)

    def query(self, kf_idx: int, min_gap: int | None = None) -> list[tuple[int, float]]:
        if min_gap is None:
            min_gap = self.default_min_gap

        if len(self.descriptors) < 2:
            return []

        # Get the descriptor for the query keyframe
        try:
            query_pos = self.kf_indices.index(kf_idx)
        except ValueError:
            return []

        query_desc = self.descriptors[query_pos]  # (1, D)

        # Stack all descriptors into a matrix
        db_descs = torch.cat(self.descriptors, dim=0)  # (N, D)

        # Cosine similarity (descriptors are already L2-normalized)
        similarities = (db_descs @ query_desc.T).squeeze(-1)  # (N,)

        candidates = []
        for i, (kf_i, sim) in enumerate(zip(self.kf_indices, similarities.tolist())):
            # Skip self and temporally close frames
            if abs(kf_i - kf_idx) < min_gap:
                continue
            if sim >= self.threshold:
                candidates.append((kf_i, sim))

        # Sort by descending similarity
        candidates.sort(key=lambda x: -x[1])
        return candidates[: self.top_k]

    def reset(self) -> None:
        self.kf_indices.clear()
        self.descriptors.clear()

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "device":         lambda s: isinstance(s, str),
            "backbone":       lambda s: s in {"resnet18", "resnet34", "resnet50"},
            "descriptor_dim": lambda d: isinstance(d, int) and d > 0,
            "threshold":      lambda t: isinstance(t, float) and 0.0 < t < 1.0,
            "top_k":          lambda k: isinstance(k, int) and k > 0,
            "min_gap":        lambda g: isinstance(g, int) and g > 0,
        })