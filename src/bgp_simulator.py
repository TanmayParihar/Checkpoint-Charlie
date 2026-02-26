"""
BGP Route Simulator
===================
Generates realistic BGP UPDATE announcements by computing paths through the
AS topology and recording route metadata needed by all validators.

AS_PATH format used throughout this project
--------------------------------------------
  [receiving_peer_as, ..., intermediate_ases, ..., origin_as]

  •  as_path[-1]  = origin AS (announced the prefix)
  •  as_path[-2]  = first AS that relayed from origin
  •  as_path[0]   = the AS that just sent the UPDATE to us (our peer)

This mirrors the BGP wire format where each AS prepends itself.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.topology import ASTopology


@dataclass
class BGPRoute:
    """A single BGP route announcement."""
    prefix:      str            # IP prefix (e.g. "10.0.0.0/16")
    origin_as:   int            # True origin AS (rightmost in AS_PATH)
    as_path:     List[int]      # Full AS_PATH as described above
    timestamp:   float = field(default_factory=time.time)

    # Ground-truth attack metadata (unknown to validators)
    is_attack:   bool = False
    attack_type: Optional[str] = None   # "origin_hijack" | "path_shortening" | "path_fabrication"
    attacker_as: Optional[int] = None

    @property
    def path_length(self) -> int:
        return len(self.as_path)

    def __repr__(self) -> str:
        tag = f" [{self.attack_type}]" if self.is_attack else ""
        return f"BGPRoute({self.prefix}, path={self.as_path}{tag})"


class BGPSimulator:
    """
    Generates a stream of BGP route announcements.

    Parameters
    ----------
    topology : ASTopology
        Pre-built AS topology.
    num_routes : int
        Total number of announcements to generate (including attacks).
    attack_rate : float
        Fraction of announcements that are attacks (injected later).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        topology: ASTopology,
        num_routes: int = 20_000,
        seed: int = 42,
    ) -> None:
        self.topology   = topology
        self.num_routes = num_routes
        self.rng        = random.Random(seed)
        self.np_rng     = np.random.default_rng(seed)

        # Historical tracking: prefix → list of observed (origin_as, as_path)
        self._history: Dict[str, List[Tuple[int, List[int]]]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_normal_routes(self, n: Optional[int] = None) -> List[BGPRoute]:
        """
        Generate *n* valid (non-attack) BGP route announcements by computing
        valley-free-ish paths through the topology graph.
        """
        n = n or self.num_routes
        routes: List[BGPRoute] = []

        all_ases = list(self.topology.graph.nodes())
        # Pre-compute candidate (src, dst) pairs
        pairs = self.topology.sample_as_pairs(min(n * 2, 50_000))
        pair_iter = iter(pairs)

        with tqdm(total=n, desc="Generating normal routes", unit="route") as pbar:
            while len(routes) < n:
                try:
                    src, origin_as = next(pair_iter)
                except StopIteration:
                    # Refill pairs if needed
                    pairs = self.topology.sample_as_pairs(n)
                    pair_iter = iter(pairs)
                    src, origin_as = next(pair_iter)

                # Pick a random prefix owned by the origin AS
                if origin_as not in self.topology.prefixes or not self.topology.prefixes[origin_as]:
                    continue
                prefix = self.rng.choice(self.topology.prefixes[origin_as])

                # Compute path: BGP path goes from src toward origin_as
                path = self.topology.get_path(src, origin_as)
                if path is None or len(path) < 2 or len(path) > 8:
                    continue

                route = BGPRoute(
                    prefix=prefix,
                    origin_as=origin_as,
                    as_path=path,       # [src, ..., origin_as]
                )
                routes.append(route)

                # Record in history for anomaly detector training
                if prefix not in self._history:
                    self._history[prefix] = []
                self._history[prefix].append((origin_as, list(path)))

                pbar.update(1)

        return routes

    def get_history(self) -> Dict[str, List[Tuple[int, List[int]]]]:
        """Return the historical route record (used to train anomaly detector)."""
        return dict(self._history)

    def path_length_stats(self, routes: List[BGPRoute]) -> Dict:
        """Compute basic path-length statistics over a route set."""
        lengths = [r.path_length for r in routes]
        return {
            "mean":   float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std":    float(np.std(lengths)),
            "min":    int(np.min(lengths)),
            "max":    int(np.max(lengths)),
        }
