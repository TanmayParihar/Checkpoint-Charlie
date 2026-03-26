"""
Path Plausibility Scoring Engine
================================
Fuses heterogeneous BGP validation signals into a single plausibility score.

Output
------
- score:   continuous [0.0, 1.0], where 1.0 is highly plausible
- verdict: "accept" | "suspect" | "reject"
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import config
from src.bgp_simulator import BGPRoute
from src.validators.base import BaseValidator


class PathPlausibilityValidator(BaseValidator):
    """Weighted Bayesian fusion across multiple validator signals."""

    name = "Path Plausibility"

    DEFAULT_WEIGHTS = {
        "rpki": 1.20,
        "aspa": 1.35,
        "otc": 0.90,
        "selective": 1.20,
        "anomaly": 0.95,
        "gnn": 1.45,
        "delta": 0.80,
    }

    SIGNAL_NAMES = (
        "rpki",
        "aspa",
        "otc",
        "selective",
        "anomaly",
        "gnn",
        "delta",
    )

    def __init__(
        self,
        rpki_validator,
        aspa_validator,
        otc_validator,
        selective_validator,
        anomaly_validator,
        gnn_validator,
        delta_validator,
        *,
        weights: Optional[Dict[str, float]] = None,
        enabled_signals: Optional[Iterable[str]] = None,
        accept_threshold: float = 0.70,
        suspect_threshold: float = 0.45,
        reject_suspect: bool = True,
    ) -> None:
        self.rpki_validator = rpki_validator
        self.aspa_validator = aspa_validator
        self.otc_validator = otc_validator
        self.selective_validator = selective_validator
        self.anomaly_validator = anomaly_validator
        self.gnn_validator = gnn_validator
        self.delta_validator = delta_validator

        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        if enabled_signals is None:
            self.enabled_signals: Set[str] = set(self.SIGNAL_NAMES)
        else:
            self.enabled_signals = set(enabled_signals)

        self.accept_threshold = accept_threshold
        self.suspect_threshold = suspect_threshold
        self.reject_suspect = reject_suspect

        self.last_components: Dict[str, float] = {}
        self.last_verdict: str = "unknown"

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        score, verdict, _, t_ms = self.score_route(route)
        if verdict == "accept":
            return True, t_ms
        if verdict == "suspect":
            return (not self.reject_suspect), t_ms
        return False, t_ms

    def score_route(self, route: BGPRoute) -> Tuple[float, str, Dict[str, float], float]:
        """
        Return (score, verdict, component_probabilities, processing_time_ms).
        """
        component_prob: Dict[str, float] = {}
        t_total = 0.0

        if "rpki" in self.enabled_signals:
            prob, t = self._rpki_signal(route)
            component_prob["rpki"] = prob
            t_total += t

        if "aspa" in self.enabled_signals:
            prob, t = self._aspa_signal(route)
            component_prob["aspa"] = prob
            t_total += t

        if "otc" in self.enabled_signals:
            prob, t = self._otc_signal(route)
            component_prob["otc"] = prob
            t_total += t

        if "selective" in self.enabled_signals:
            prob, t = self._selective_signal(route)
            component_prob["selective"] = prob
            t_total += t

        if "anomaly" in self.enabled_signals:
            prob, t = self._anomaly_signal(route)
            component_prob["anomaly"] = prob
            t_total += t

        if "gnn" in self.enabled_signals:
            prob, t = self._gnn_signal(route)
            component_prob["gnn"] = prob
            t_total += t

        if "delta" in self.enabled_signals:
            prob, t = self._delta_signal(route)
            component_prob["delta"] = prob
            t_total += t

        score = self._fuse(component_prob)
        verdict = self._categorize(score)

        self.last_components = component_prob
        self.last_verdict = verdict
        # fusion arithmetic overhead
        t_total += 0.01
        return score, verdict, component_prob, t_total

    # ---- Signal extraction ---------------------------------------------

    def _rpki_signal(self, route: BGPRoute) -> Tuple[float, float]:
        claimed_origin = route.as_path[-1] if route.as_path else route.origin_as
        authorised = self.rpki_validator.roa_db.get(route.prefix)
        t_ms = config.HASH_LOOKUP_MS
        if authorised is None:
            return 0.55, t_ms
        t_ms += config.CRYPTO_VERIFY_MS
        if claimed_origin == authorised:
            return 0.96, t_ms
        return 0.02, t_ms

    def _aspa_signal(self, route: BGPRoute) -> Tuple[float, float]:
        if hasattr(self.aspa_validator, "verify_state"):
            state, t_ms = self.aspa_validator.verify_state(route)
        else:
            ok, t_ms = self.aspa_validator.validate(route)
            state = "valid" if ok else "invalid"
        if state == "valid":
            return 0.94, t_ms
        if state == "invalid":
            return 0.03, t_ms
        return 0.55, t_ms

    def _otc_signal(self, route: BGPRoute) -> Tuple[float, float]:
        if hasattr(self.otc_validator, "check_state"):
            state, t_ms = self.otc_validator.check_state(route)
        else:
            ok, t_ms = self.otc_validator.validate(route)
            state = "valid" if ok else "invalid"
        if state == "valid":
            return 0.84, t_ms
        if state == "invalid":
            return 0.08, t_ms
        return 0.55, t_ms

    def _selective_signal(self, route: BGPRoute) -> Tuple[float, float]:
        ok, t_ms = self.selective_validator.validate(route)
        return (0.87 if ok else 0.10), t_ms

    def _anomaly_signal(self, route: BGPRoute) -> Tuple[float, float]:
        if hasattr(self.anomaly_validator, "_compute_score"):
            raw = float(self.anomaly_validator._compute_score(route))
            _, t_ms = self.anomaly_validator.validate(route)
        else:
            ok, t_ms = self.anomaly_validator.validate(route)
            raw = 0.1 if ok else 0.9
        # Convert anomaly score to plausibility.
        return self._clamp(1.0 - raw), t_ms

    def _gnn_signal(self, route: BGPRoute) -> Tuple[float, float]:
        if hasattr(self.gnn_validator, "score_route"):
            raw = float(self.gnn_validator.score_route(route))
            if hasattr(self.gnn_validator, "inference_time_ms"):
                t_ms = float(self.gnn_validator.inference_time_ms())
            else:
                _, t_ms = self.gnn_validator.validate(route)
        else:
            ok, t_ms = self.gnn_validator.validate(route)
            raw = 0.1 if ok else 0.9
        return self._clamp(1.0 - raw), t_ms

    def _delta_signal(self, route: BGPRoute) -> Tuple[float, float]:
        path = route.as_path
        if len(path) < 2:
            ok, t_ms = self.delta_validator.validate(route)
            return (0.55 if ok else 0.05), t_ms

        edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
        cached = self.delta_validator._trust_cache.get(route.prefix, set())
        cache_hit_ratio = len(edges & cached) / max(1, len(edges))

        ok, t_ms = self.delta_validator.validate(route)
        if not ok:
            return 0.05, t_ms
        return self._clamp(0.55 + 0.40 * cache_hit_ratio), t_ms

    # ---- Fusion ---------------------------------------------------------

    def _fuse(self, component_prob: Dict[str, float]) -> float:
        if not component_prob:
            return 0.5

        odds = 1.0
        for signal, prob in component_prob.items():
            w = self.weights.get(signal, 1.0)
            p = self._clamp(prob)
            odds *= (p / (1.0 - p)) ** w
        return odds / (1.0 + odds)

    def _categorize(self, score: float) -> str:
        if score >= self.accept_threshold:
            return "accept"
        if score >= self.suspect_threshold:
            return "suspect"
        return "reject"

    @staticmethod
    def _clamp(p: float, eps: float = 1e-2) -> float:
        return min(max(p, eps), 1.0 - eps)
