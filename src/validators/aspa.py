"""
ASPA Validator
==============
Implements a lightweight ASPA-style path authorization check with partial
deployment support.

Validation states
-----------------
- valid:   covered customer->provider links are authorized
- invalid: at least one covered customer->provider link is unauthorized
- unknown: no ASPA-covered customer appeared on the path

The boolean validator contract maps these states to:
- valid/unknown -> accepted (True)
- invalid       -> rejected (False)
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology, PEER, PROVIDER
from src.validators.base import BaseValidator


class ASPAValidator(BaseValidator):
    """ASPA-style AS_PATH verification with configurable deployment coverage."""

    name = "ASPA"

    def __init__(
        self,
        topology: ASTopology,
        coverage: float = 0.50,
        seed: int = 42,
        allow_peer_links: bool = True,
    ) -> None:
        self.topology = topology
        self.coverage = max(0.0, min(1.0, coverage))
        self.allow_peer_links = allow_peer_links
        self.rng = random.Random(seed)

        # customer_as -> authorized provider ASNs
        self.aspa_db: Dict[int, Set[int]] = {}
        self._build_aspa_db()

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        state, t_ms = self.verify_state(route)
        return state != "invalid", t_ms

    def verify_state(self, route: BGPRoute) -> Tuple[str, float]:
        """
        Return (state, processing_time_ms), where state is:
        "valid" | "invalid" | "unknown".
        """
        path = route.as_path
        if len(path) < 2:
            return "unknown", config.HASH_LOOKUP_MS

        t_ms = config.HASH_LOOKUP_MS
        covered_checks = 0

        # Path is [receiver, ..., origin]. We inspect each adjacent pair.
        for i in range(len(path) - 1):
            left = path[i]
            right = path[i + 1]

            rel_left_right = self.topology.relationship(left, right)
            rel_right_left = self.topology.relationship(right, left)

            # Non-adjacent/unknown relation => path inconsistency.
            if rel_left_right is None or rel_right_left is None:
                return "invalid", t_ms + config.HASH_LOOKUP_MS

            # Upstream-style check: if RIGHT sees LEFT as provider,
            # RIGHT must authorize LEFT in ASPA (if RIGHT is covered).
            if rel_right_left == PROVIDER:
                t_ms += config.HASH_LOOKUP_MS
                providers = self.aspa_db.get(right)
                if providers is not None:
                    covered_checks += 1
                    if left not in providers:
                        return "invalid", t_ms
            elif rel_right_left == PEER and (not self.allow_peer_links):
                return "invalid", t_ms

            # Downstream-style symmetric check.
            if rel_left_right == PROVIDER:
                t_ms += config.HASH_LOOKUP_MS
                providers = self.aspa_db.get(left)
                if providers is not None:
                    covered_checks += 1
                    if right not in providers:
                        return "invalid", t_ms
            elif rel_left_right == PEER and (not self.allow_peer_links):
                return "invalid", t_ms

        if covered_checks > 0:
            return "valid", t_ms
        return "unknown", t_ms

    def coverage_stats(self) -> Dict[str, int]:
        """Return simple ASPA deployment statistics."""
        covered = len(self.aspa_db)
        total = self.topology.graph.number_of_nodes()
        return {
            "covered_ases": covered,
            "total_ases": total,
            "coverage_pct": int(round(100.0 * covered / max(1, total))),
        }

    def _build_aspa_db(self) -> None:
        for asn in self.topology.graph.nodes():
            if self.rng.random() > self.coverage:
                continue

            providers: Set[int] = set()
            for nbr in self.topology.graph.neighbors(asn):
                rel = self.topology.relationship(asn, nbr)
                if rel == PROVIDER:
                    providers.add(nbr)

            # Covered ASes with no provider links still publish an empty set.
            self.aspa_db[asn] = providers
