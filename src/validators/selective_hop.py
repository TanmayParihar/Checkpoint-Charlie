"""
Selective Hop Validator
=======================
The core contribution of this project.  Instead of verifying every AS hop
(BGPsec), this validator checks only *selected* positions in the AS_PATH.

Position semantics
------------------
  as_path = [peer, hop_n-2, ..., hop_1, origin]

  Position index (our convention, 0-based from origin):
    0   → origin           (as_path[-1])
    1   → 1st hop from origin (as_path[-2])
    2   → 2nd hop from origin (as_path[-3])
    …
    -1  → receiving peer   (as_path[0])
    -2  → 2nd from receiver (as_path[1])

Cryptographic model
-------------------
Each AS along a legitimate path is assumed to have signed its adjacency with
the next hop.  Verification is modelled as follows:

  • For each verified position p, we check that as_path[p] and as_path[p+1]
    (its neighbour toward the receiver) are topologically adjacent in the AS
    graph.  A non-adjacent pair signals a tampered or forged hop.
  • Timing: len(verified_positions) × CRYPTO_VERIFY_MS
  • For paths shorter than what the position set requires, the validator
    degrades gracefully (only checks positions that exist).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology
from src.validators.base import BaseValidator


class SelectiveHopValidator(BaseValidator):
    """
    Validates a configurable subset of AS_PATH hops.

    Parameters
    ----------
    topology : ASTopology
        Used to check physical adjacency between ASes.
    positions : list[int] or None
        Which positions to verify (see module docstring).
        None means verify ALL hops (equivalent to BGPsec).
    label : str
        Human-readable name shown in reports.
    """

    def __init__(
        self,
        topology: ASTopology,
        positions: Optional[List[int]] = None,
        label: str = "SelectiveHop",
    ) -> None:
        self.topology  = topology
        self.positions = positions   # None → full path (BGPsec)
        self.name      = label

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        """
        Check topology-adjacency for the selected hop positions.

        Returns (True, t) if all selected hops are valid, (False, t) otherwise.
        """
        path = route.as_path
        n    = len(path)

        if n < 2:
            return True, config.HASH_LOOKUP_MS

        # Determine which positions to verify
        if self.positions is None:
            # Full path: verify every consecutive pair
            check_indices = list(range(n - 1))   # indices into path
        else:
            check_indices = self._resolve_positions(path)

        t = len(check_indices) * config.CRYPTO_VERIFY_MS

        # Check each selected adjacency
        for idx in check_indices:
            if idx + 1 >= n:
                continue   # boundary guard
            as_a = path[idx]
            as_b = path[idx + 1]
            if not self.topology.are_adjacent(as_a, as_b):
                return False, t   # adjacency broken → tampered

        return True, t

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_positions(self, path: List[int]) -> List[int]:
        """
        Convert our position conventions to actual indices in `path`.

        Position p means:
          p >= 0 → checking adjacency starting from the *right* (origin side)
                   path[-1] is origin, so we check path[-1-p] ↔ path[-p]
          p < 0  → checking from the *left* (receiver side)
                   e.g. -1 checks path[0] ↔ path[1]

        We return the LEFT index of each adjacency pair (path[idx] ↔ path[idx+1]).
        """
        n = len(path)
        indices: List[int] = []

        for p in self.positions:
            if p >= 0:
                # Origin-relative: origin is at index n-1
                # The adjacency "at position p from origin" is between
                # path[n-1-p-1] and path[n-1-p], i.e., left index = n-2-p
                left = n - 2 - p
            else:
                # Receiver-relative: receiver peer is at index 0
                # -1 → check path[0]↔path[1], left index = abs(p)-1
                left = abs(p) - 1

            if 0 <= left < n - 1:
                indices.append(left)

        # Remove duplicates and sort
        return sorted(set(indices))
