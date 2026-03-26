"""
Delta / Incremental Trust Propagation Validator
================================================
The git-diff model applied to BGP path verification.

Core idea: cache a trust graph of previously verified path segments (edges).
When a new announcement arrives for a prefix we have seen before, compute the
diff from the last known-good path.  Only verify the *changed* edges -- treat
unchanged segments as already trusted from previous verification.

Verification cost scales with path **churn**, not path **length**.  On stable
prefixes (the vast majority of Internet routing) this is nearly free after the
first verification pass.

Trust cache structure
---------------------
  _trust_cache[prefix] → set of verified (as_a, as_b) directed edges

When a route for a known prefix arrives:
  1. Extract current edges from the AS_PATH.
  2. Compute new_edges = current_edges - cached_edges.
  3. Verify only new_edges via topology adjacency check.
  4. If all pass, merge new_edges into cache → route accepted.
  5. If any fail → route rejected (tampered).

Timing model
------------
  • Cache lookup: HASH_LOOKUP_MS
  • Per new edge: CRYPTO_VERIFY_MS  (same as selective-hop / BGPsec)
  • Unchanged edges: 0 cost (trusted from cache)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology
from src.validators.base import BaseValidator


class DeltaTrustValidator(BaseValidator):
    """
    Incremental trust propagation validator.

    Caches verified path edges per prefix.  On subsequent announcements for the
    same prefix, only the edges that changed since the last verification are
    cryptographically verified.

    Parameters
    ----------
    topology : ASTopology
        Used to check physical adjacency between ASes.
    label : str
        Human-readable name shown in reports.
    """

    name = "Delta Trust"

    def __init__(
        self,
        topology: ASTopology,
        label: str = "Delta Trust",
    ) -> None:
        self.topology = topology
        self.name = label

        # prefix → set of previously verified directed edges (as_a, as_b)
        self._trust_cache: Dict[str, Set[Tuple[int, int]]] = {}

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        """
        Validate via incremental trust propagation.

        Returns (True, t) if all new edges are valid, (False, t) otherwise.
        """
        path = route.as_path
        prefix = route.prefix

        if len(path) < 2:
            return True, config.HASH_LOOKUP_MS

        # Extract directed edges from current AS_PATH
        current_edges: Set[Tuple[int, int]] = set()
        for i in range(len(path) - 1):
            current_edges.add((path[i], path[i + 1]))

        # Cache lookup cost
        t = config.HASH_LOOKUP_MS

        # Retrieve cached (already-verified) edges for this prefix
        cached_edges = self._trust_cache.get(prefix, set())

        # Delta: only the edges we haven't verified before
        new_edges = current_edges - cached_edges

        # Verification cost scales with churn, not path length
        t += len(new_edges) * config.CRYPTO_VERIFY_MS

        # Verify each new edge
        for as_a, as_b in new_edges:
            if not self.topology.are_adjacent(as_a, as_b):
                # Tampered edge detected -- do NOT cache bad edges
                return False, t

        # All new edges verified successfully → update cache
        self._trust_cache[prefix] = cached_edges | current_edges

        return True, t

    def reset_cache(self) -> None:
        """Clear the trust cache (useful between simulation runs)."""
        self._trust_cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """Return basic statistics about the trust cache."""
        total_edges = sum(len(edges) for edges in self._trust_cache.values())
        return {
            "prefixes_cached": len(self._trust_cache),
            "total_cached_edges": total_edges,
        }
