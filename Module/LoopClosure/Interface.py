import torch
import pypose as pp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace

from Utility.Extensions import ConfigTestableSubclass


@dataclass
class LoopClosureResult:
    """Result of a successful loop closure detection + geometric verification."""
    query_kf_idx:   int                # Keyframe index of query (current)
    match_kf_idx:   int                # Keyframe index of matched (historical)
    relative_pose:  pp.LieTensor       # T_ab: relative SE3 from query to match  (1x7)
    covariance:     torch.Tensor       # 6x6 marginalized covariance on se(3)
    confidence:     float              # Similarity / inlier ratio


class ILoopClosureDetector(ABC, ConfigTestableSubclass):
    """
    Interface for loop closure detection, analogous to IMatcher.
    
    Workflow:
        1. add_keyframe(kf_idx, image) — encode and store descriptor
        2. query(kf_idx) — find candidate matches (VPR retrieval)
        3. Geometric verification is done externally (using MAC-VO frontend)
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config

    @abstractmethod
    def add_keyframe(self, kf_idx: int, image: torch.Tensor) -> None:
        """
        Encode keyframe image and add to the descriptor database.
        Args:
            kf_idx: Global keyframe index in the VisualMap
            image:  (1, C, H, W) tensor, the left camera image
        """
        ...

    @abstractmethod
    def query(self, kf_idx: int, min_gap: int = 30) -> list[tuple[int, float]]:
        """
        Query the database for loop closure candidates.
        Args:
            kf_idx:   Current keyframe index
            min_gap:  Minimum temporal gap between query and candidate to avoid
                      matching sequential frames.
        Returns:
            List of (candidate_kf_idx, similarity_score), sorted by descending score.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear the descriptor database."""
        ...