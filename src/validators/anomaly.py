"""
Statistical Anomaly Detector
=============================
A cryptography-free, statistics-based first-line defence that scores each
BGP route announcement against a learned model of normal routing behaviour.

Detection signals
-----------------
1. **Path length z-score** – Is this path significantly shorter/longer than
   what we normally see for this origin AS?

2. **Origin consistency** – Has this prefix been announced from a *different*
   origin AS before?  Sudden origin changes are a strong indicator of hijacks.

3. **Valley-free violation** – Does the AS_PATH violate the valley-free
   routing constraint?  Fabricated or shortened paths often do.

4. **New AS appearance** – Does the path contain ASes we have never seen in
   *any* historical route?  Rare but present ASes are weakly suspicious; a
   completely novel AS is strongly suspicious.

5. **AS repetition (loop detection)** – Legitimate paths never repeat an AS
   number.  A duplicate indicates a spoofed or looped path.

Scoring
-------
Each signal contributes a partial score in [0, 1].  The final anomaly score
is a weighted sum clamped to [0, 1].  Routes with score > threshold are
flagged as anomalous (= rejected).

The detector must be **trained** on normal routes *before* being used to
validate traffic (call `train(normal_routes)` first).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology
from src.validators.base import BaseValidator


class AnomalyDetector(BaseValidator):
    """
    Statistical anomaly-based BGP route validator.

    Parameters
    ----------
    topology : ASTopology
        Used for valley-free check.
    threshold : float
        Anomaly score cutoff (0–1).  Routes above this are flagged.
    """

    name = "Anomaly Detector"

    # ── Weights for each detection signal (must sum to ≤1 each individually) ─
    W_PATH_LENGTH   = 0.25
    W_ORIGIN_CHANGE = 0.35
    W_VALLEY_FREE   = 0.20
    W_NEW_AS        = 0.10
    W_LOOP          = 0.10

    def __init__(
        self,
        topology: ASTopology,
        threshold: float = config.ANOMALY_SCORE_THRESHOLD,
    ) -> None:
        self.topology  = topology
        self.threshold = threshold

        # Trained model state (populated by train())
        self._trained = False

        # Per-origin AS: (mean path length, std path length)
        self._origin_path_stats: Dict[int, Tuple[float, float]] = {}

        # prefix → set of historically seen origin ASes
        self._prefix_origins: Dict[str, Set[int]] = defaultdict(set)

        # Set of all ASes ever seen in any path (universe of known ASes)
        self._known_ases: Set[int] = set()

        # Global path length stats (fallback when per-origin stats absent)
        self._global_mean_len: float = 4.0
        self._global_std_len: float  = 1.5

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, normal_routes: List[BGPRoute]) -> None:
        """
        Build the baseline model from a list of known-legitimate routes.
        Call this *before* running validate().
        """
        # Collect path lengths per origin AS
        lengths_by_origin: Dict[int, List[int]] = defaultdict(list)
        all_lengths: List[int] = []

        for route in normal_routes:
            origin = route.as_path[-1] if route.as_path else route.origin_as
            plen   = route.path_length

            lengths_by_origin[origin].append(plen)
            all_lengths.append(plen)

            self._prefix_origins[route.prefix].add(origin)
            self._known_ases.update(route.as_path)

        # Fit per-origin stats
        for origin, lengths in lengths_by_origin.items():
            arr = np.array(lengths, dtype=float)
            self._origin_path_stats[origin] = (
                float(np.mean(arr)),
                max(float(np.std(arr)), 0.5),   # minimum std=0.5 to avoid /0
            )

        # Global fallback stats
        if all_lengths:
            arr = np.array(all_lengths, dtype=float)
            self._global_mean_len = float(np.mean(arr))
            self._global_std_len  = max(float(np.std(arr)), 0.5)

        self._trained = True

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        """
        Compute an anomaly score for the route and return
        (not_anomalous, processing_time_ms).
        """
        if not self._trained:
            raise RuntimeError("AnomalyDetector.train() must be called before validate().")

        # Cost = a few hash lookups (no crypto)
        t = config.HASH_LOOKUP_MS * 5

        score = self._compute_score(route)
        is_valid = score <= self.threshold
        return is_valid, t

    # ── Score components ──────────────────────────────────────────────────────

    def _compute_score(self, route: BGPRoute) -> float:
        score = 0.0
        score += self.W_PATH_LENGTH   * self._score_path_length(route)
        score += self.W_ORIGIN_CHANGE * self._score_origin_change(route)
        score += self.W_VALLEY_FREE   * self._score_valley_free(route)
        score += self.W_NEW_AS        * self._score_new_as(route)
        score += self.W_LOOP          * self._score_loop(route)
        return min(score, 1.0)

    def _score_path_length(self, route: BGPRoute) -> float:
        """
        Return a score in [0,1] proportional to how unusual the path length is.
        Uses z-score capped at Z_LIMIT.
        """
        origin = route.as_path[-1] if route.as_path else route.origin_as
        if origin in self._origin_path_stats:
            mean, std = self._origin_path_stats[origin]
        else:
            mean, std = self._global_mean_len, self._global_std_len

        z = abs(route.path_length - mean) / std
        limit = config.PATH_LENGTH_ZSCORE_LIMIT
        return min(z / limit, 1.0)

    def _score_origin_change(self, route: BGPRoute) -> float:
        """
        Return 1.0 if the origin AS has changed from all previously seen
        origins for this prefix, 0.0 if origin is consistent.
        """
        claimed_origin = route.as_path[-1] if route.as_path else route.origin_as
        seen_origins   = self._prefix_origins.get(route.prefix, set())
        if not seen_origins:
            # Never seen this prefix → mildly suspicious (new prefix)
            return 0.3
        if claimed_origin not in seen_origins:
            return 1.0    # origin changed → very suspicious
        return 0.0

    def _score_valley_free(self, route: BGPRoute) -> float:
        """Return 1.0 if the path violates valley-free, 0.0 otherwise."""
        return 0.0 if self.topology.is_valley_free(route.as_path) else 1.0

    def _score_new_as(self, route: BGPRoute) -> float:
        """
        Fraction of ASes in the path that have never been seen before.
        """
        if not route.as_path:
            return 0.0
        unknown = sum(1 for asn in route.as_path if asn not in self._known_ases)
        return unknown / len(route.as_path)

    @staticmethod
    def _score_loop(route: BGPRoute) -> float:
        """Return 1.0 if any AS appears twice in the path (loop), else 0.0."""
        return 0.0 if len(route.as_path) == len(set(route.as_path)) else 1.0
