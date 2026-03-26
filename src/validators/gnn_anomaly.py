"""
Topology-Aware GNN-Style Anomaly Detector
=========================================
This module provides a graph-aware anomaly detector that uses AS-topology
message passing to produce route embeddings and anomaly scores.

Design notes
------------
- Uses two rounds of neighborhood propagation (GNN-style message passing)
  over AS node features.
- Produces a per-route anomaly score in [0, 1].
- Supports a lightweight CPU path by default, with optional "torch" engine
  selection for timing-model purposes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology, PEER
from src.validators.base import BaseValidator


class GNNAnomalyDetector(BaseValidator):
    """
    Graph-aware anomaly detector with a 2-layer message-passing backbone.

    Parameters
    ----------
    topology : ASTopology
        AS-level graph and relationship metadata.
    threshold : float
        Routes with score > threshold are flagged as anomalous.
    engine : str
        "auto" | "cpu" | "torch" (affects timing model only).
    """

    name = "GNN Anomaly Detector"

    def __init__(
        self,
        topology: ASTopology,
        threshold: float = 0.58,
        engine: str = "auto",
    ) -> None:
        self.topology = topology
        self.threshold = threshold
        self.engine = self._resolve_engine(engine)

        self._trained = False
        self._asn_to_idx: Dict[int, int] = {}
        self._idx_to_asn: List[int] = []
        self._neighbors: List[List[Tuple[int, float]]] = []

        self._node_features: Optional[np.ndarray] = None
        self._node_embeddings: Optional[np.ndarray] = None

        self._mean_vec: Optional[np.ndarray] = None
        self._std_vec: Optional[np.ndarray] = None

    # ---- Public API -----------------------------------------------------

    def train(self, normal_routes: Sequence[BGPRoute]) -> None:
        """Fit baseline embedding statistics from known-legitimate traffic."""
        self._index_ases()
        self._neighbors = self._build_neighbor_cache()
        self._node_features = self._build_node_features(normal_routes)
        self._node_embeddings = self._run_message_passing(self._node_features)

        vectors = np.array([self._route_vector(r) for r in normal_routes], dtype=float)
        if vectors.size == 0:
            # Defensive fallback for degenerate training sets.
            self._mean_vec = np.zeros(9, dtype=float)
            self._std_vec = np.ones(9, dtype=float)
            self._trained = True
            return

        self._mean_vec = vectors.mean(axis=0)
        self._std_vec = np.maximum(vectors.std(axis=0), 1e-3)

        # Calibrate threshold to keep normal-route reject rate low.
        normal_scores = np.array([self._score_vector(v) for v in vectors], dtype=float)
        p97 = float(np.quantile(normal_scores, 0.97))
        self.threshold = max(self.threshold, p97)
        self._trained = True

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        if not self._trained:
            raise RuntimeError("GNNAnomalyDetector.train() must be called before validate().")
        score = self.score_route(route)
        is_valid = score <= self.threshold
        return is_valid, self.inference_time_ms()

    def score_route(self, route: BGPRoute) -> float:
        """Return anomaly score in [0, 1] (higher = more suspicious)."""
        if not self._trained:
            raise RuntimeError("GNNAnomalyDetector.train() must be called before score_route().")
        vec = self._route_vector(route)
        return self._score_vector(vec)

    def inference_time_ms(self) -> float:
        """
        Modeled inference cost:
        - torch engine: GPU-like range
        - cpu engine: CPU-like range
        """
        if self.engine == "torch":
            return 1.5
        return 6.0

    def roc_inputs(self, routes: Sequence[BGPRoute]) -> Tuple[List[int], List[float]]:
        """Convenience helper for ROC generation."""
        y_true: List[int] = []
        y_score: List[float] = []
        for route in routes:
            y_true.append(1 if route.is_attack else 0)
            y_score.append(self.score_route(route))
        return y_true, y_score

    # ---- Core scoring ---------------------------------------------------

    def _score_vector(self, vec: np.ndarray) -> float:
        assert self._mean_vec is not None
        assert self._std_vec is not None
        z = np.abs((vec - self._mean_vec) / self._std_vec)
        distance = math.sqrt(float(np.mean(z * z)))
        score = 1.0 - math.exp(-distance)
        return float(min(max(score, 0.0), 1.0))

    def _route_vector(self, route: BGPRoute) -> np.ndarray:
        assert self._node_embeddings is not None

        path = route.as_path
        indices: List[int] = []
        unknown = 0
        for asn in path:
            idx = self._asn_to_idx.get(asn)
            if idx is None:
                unknown += 1
            else:
                indices.append(idx)

        if indices:
            node_vec = self._node_embeddings[indices].mean(axis=0)
        else:
            node_vec = np.zeros(4, dtype=float)

        path_len_norm = min(len(path) / 10.0, 1.0)
        valley_free = 1.0 if self.topology.is_valley_free(path) else 0.0
        unique_ratio = (len(set(path)) / len(path)) if path else 0.0
        tier_smoothness = 1.0 - self._tier_transition_score(path)
        unknown_ratio = (unknown / len(path)) if path else 0.0

        return np.concatenate(
            [
                node_vec,
                np.array(
                    [
                        path_len_norm,
                        valley_free,
                        unique_ratio,
                        tier_smoothness,
                        unknown_ratio,
                    ],
                    dtype=float,
                ),
            ],
            axis=0,
        )

    # ---- GNN-style embedding -------------------------------------------

    def _index_ases(self) -> None:
        self._idx_to_asn = sorted(self.topology.graph.nodes())
        self._asn_to_idx = {asn: idx for idx, asn in enumerate(self._idx_to_asn)}

    def _build_node_features(self, normal_routes: Sequence[BGPRoute]) -> np.ndarray:
        announce_count: Dict[int, int] = defaultdict(int)
        prefix_diversity: Dict[int, set] = defaultdict(set)

        for route in normal_routes:
            for asn in route.as_path:
                announce_count[asn] += 1
                prefix_diversity[asn].add(route.prefix)

        max_degree = max((self.topology.graph.degree(asn) for asn in self._idx_to_asn), default=1)
        max_announce = max(announce_count.values(), default=1)
        max_prefixes = max((len(v) for v in prefix_diversity.values()), default=1)

        feats: List[List[float]] = []
        for asn in self._idx_to_asn:
            degree = self.topology.graph.degree(asn) / max(1, max_degree)
            tier = self.topology.as_tier.get(asn, 3)
            tier_score = {1: 1.0, 2: 0.6, 3: 0.2}.get(tier, 0.2)
            history = announce_count.get(asn, 0) / max(1, max_announce)
            diversity = len(prefix_diversity.get(asn, set())) / max(1, max_prefixes)
            feats.append([degree, tier_score, history, diversity])

        return np.array(feats, dtype=float)

    def _build_neighbor_cache(self) -> List[List[Tuple[int, float]]]:
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in self._idx_to_asn]
        for asn in self._idx_to_asn:
            idx = self._asn_to_idx[asn]
            bucket: List[Tuple[int, float]] = []
            for nbr in self.topology.graph.neighbors(asn):
                rel = self.topology.relationship(asn, nbr)
                weight = 0.8 if rel == PEER else 1.0
                nbr_idx = self._asn_to_idx[nbr]
                bucket.append((nbr_idx, weight))
            neighbors[idx] = bucket
        return neighbors

    def _run_message_passing(self, base_features: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        # Two rounds of neighborhood aggregation (2-layer graph backbone).
        h = base_features.copy()
        h = self._propagate_once(h, alpha=alpha)
        h = self._propagate_once(h, alpha=alpha)
        return h

    def _propagate_once(self, h: np.ndarray, alpha: float) -> np.ndarray:
        out = np.zeros_like(h, dtype=float)
        for idx in range(h.shape[0]):
            neigh = self._neighbors[idx]
            if not neigh:
                out[idx] = h[idx]
                continue

            weights = np.array([w for _, w in neigh], dtype=float)
            vecs = np.array([h[n_idx] for n_idx, _ in neigh], dtype=float)
            agg = (vecs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
            out[idx] = np.tanh(alpha * h[idx] + (1.0 - alpha) * agg)
        return out

    # ---- Utility --------------------------------------------------------

    def _tier_transition_score(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        tiers = [self.topology.as_tier.get(asn, 3) for asn in path]
        change = 0.0
        for i in range(len(tiers) - 1):
            change += abs(tiers[i] - tiers[i + 1]) / 2.0
        return min(change / max(1, len(tiers) - 1), 1.0)

    @staticmethod
    def _resolve_engine(engine: str) -> str:
        if engine in {"cpu", "torch"}:
            return engine
        # "auto": prefer torch if available, else cpu.
        try:
            import torch  # noqa: F401

            return "torch"
        except Exception:
            return "cpu"
