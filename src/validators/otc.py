"""
OTC Route Leak Detector
=======================
RFC 9234-inspired validation using an OTC-style transitive signal.

Detection rule (simplified):
- If a route carries OTC and is received from a customer, flag as leak.

Validation states
-----------------
- valid:   OTC present and direction is acceptable
- invalid: OTC present and received from customer
- unknown: no OTC signal available (or link role unavailable)
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology, CUSTOMER, PEER, PROVIDER
from src.validators.base import BaseValidator


class OTCValidator(BaseValidator):
    """Lightweight OTC-based route-leak detector with partial adoption support."""

    name = "OTC"

    def __init__(
        self,
        topology: ASTopology,
        adoption_rate: float = 0.50,
        seed: int = 42,
    ) -> None:
        self.topology = topology
        self.adoption_rate = max(0.0, min(1.0, adoption_rate))
        self.rng = random.Random(seed)

        self._otc_enabled: Set[int] = set()
        self._build_deployment_set()

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        state, t_ms = self.check_state(route)
        return state != "invalid", t_ms

    def check_state(self, route: BGPRoute) -> Tuple[str, float]:
        """
        Return (state, processing_time_ms), where state is:
        "valid" | "invalid" | "unknown".
        """
        path = route.as_path
        if len(path) < 2:
            return "unknown", config.HASH_LOOKUP_MS

        receiver = path[0]
        sender = path[1]
        rel = self.topology.relationship(receiver, sender)
        t_ms = config.HASH_LOOKUP_MS

        if rel is None:
            return "unknown", t_ms

        otc_present = self._otc_present(route, receiver, sender, rel)
        if not otc_present:
            return "unknown", t_ms

        # RFC 9234-style leak signal:
        # OTC MUST NOT arrive from a customer.
        if rel == CUSTOMER:
            return "invalid", t_ms + config.HASH_LOOKUP_MS
        return "valid", t_ms + config.HASH_LOOKUP_MS

    def deployment_stats(self) -> Dict[str, int]:
        total = self.topology.graph.number_of_nodes()
        enabled = len(self._otc_enabled)
        return {
            "enabled_ases": enabled,
            "total_ases": total,
            "adoption_pct": int(round(100.0 * enabled / max(1, total))),
        }

    def _otc_present(self, route: BGPRoute, receiver: int, sender: int, rel: int) -> bool:
        # Explicit metadata from attack generation has priority.
        if hasattr(route, "otc_present"):
            return bool(getattr(route, "otc_present"))
        if hasattr(route, "otc"):
            return bool(getattr(route, "otc"))

        # If sender has not adopted OTC, the signal is unavailable.
        if sender not in self._otc_enabled:
            return False

        # Heuristic for synthetic traffic:
        # routes learned from provider/peer are likely to carry OTC onward.
        return rel in (PROVIDER, PEER)

    def _build_deployment_set(self) -> None:
        for asn in self.topology.graph.nodes():
            if self.rng.random() <= self.adoption_rate:
                self._otc_enabled.add(asn)
