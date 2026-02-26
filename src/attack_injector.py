"""
BGP Attack Injector
===================
Takes a set of legitimate BGP routes and injects attack variants that model
the three most common BGP path-security threats:

1. Origin Hijacking
   A malicious AS announces a prefix it doesn't own.  The attacker replaces
   (or prepends) the origin AS in the AS_PATH.

2. Path Shortening
   An AS removes 1–2 intermediate hops to make its path appear shorter and
   thus more attractive to other ASes (traffic engineering attack / MitM
   setup).

3. Path Fabrication
   An AS inserts one or more fake ASes into the middle of the path to create
   plausible-looking but false routing information (used to deceive topological
   analysis or to disguise a detour).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.bgp_simulator import BGPRoute
from src.topology import ASTopology


class AttackInjector:
    """
    Injects attack announcements into a legitimate route stream.

    Parameters
    ----------
    topology : ASTopology
        The AS topology (used to pick plausible attacker ASes).
    attack_rate : float
        Fraction of the final traffic that should be attack announcements.
    attack_type_dist : dict[str, float]
        Probabilities for each attack type (must sum to 1.0).
    seed : int
    """

    ATTACK_TYPES = ("origin_hijack", "path_shortening", "path_fabrication")

    def __init__(
        self,
        topology: ASTopology,
        attack_rate: float = 0.15,
        attack_type_dist: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        self.topology   = topology
        self.attack_rate = attack_rate
        self.rng        = random.Random(seed)
        self.np_rng     = np.random.default_rng(seed)

        if attack_type_dist is None:
            attack_type_dist = {
                "origin_hijack":    0.40,
                "path_shortening":  0.30,
                "path_fabrication": 0.30,
            }
        self.attack_dist = attack_type_dist
        self._all_ases   = list(topology.graph.nodes())

    # ── Public API ────────────────────────────────────────────────────────────

    def mix_attacks(self, normal_routes: List[BGPRoute]) -> List[BGPRoute]:
        """
        Given a list of *normal* routes return a new list that contains both
        the original routes and injected attack routes so that attack_rate of
        the total is malicious.

        The attack routes are based on modified copies of normal routes.
        """
        n_total   = len(normal_routes)
        n_attacks = int(n_total * self.attack_rate / (1 - self.attack_rate))
        n_attacks = min(n_attacks, n_total)   # cap at #normal for safety

        # Pick base routes to mutate (with replacement is fine)
        base_routes = self.rng.choices(normal_routes, k=n_attacks)

        attack_types = list(self.attack_dist.keys())
        type_probs   = [self.attack_dist[t] for t in attack_types]

        attack_routes: List[BGPRoute] = []
        with tqdm(total=n_attacks, desc="Injecting attacks", unit="attack") as pbar:
            for base in base_routes:
                a_type = self.rng.choices(attack_types, weights=type_probs, k=1)[0]
                attacked = self._inject(copy.deepcopy(base), a_type)
                if attacked is not None:
                    attack_routes.append(attacked)
                pbar.update(1)

        mixed = normal_routes + attack_routes
        self.rng.shuffle(mixed)
        return mixed

    # ── Attack implementations ────────────────────────────────────────────────

    def _inject(self, route: BGPRoute, attack_type: str) -> Optional[BGPRoute]:
        """Dispatch to the appropriate attack method."""
        if attack_type == "origin_hijack":
            return self._origin_hijack(route)
        elif attack_type == "path_shortening":
            return self._path_shortening(route)
        elif attack_type == "path_fabrication":
            return self._path_fabrication(route)
        return None

    def _origin_hijack(self, route: BGPRoute) -> BGPRoute:
        """
        Replace the origin AS (last element of as_path) with a different,
        randomly chosen attacker AS.  The prefix remains the same, so any AS
        that has a ROA for this prefix will detect the mismatch.
        """
        all_ases      = self._all_ases
        attacker_as   = route.as_path[-1]

        # Pick a different AS as the attacker
        for _ in range(20):
            candidate = self.rng.choice(all_ases)
            if candidate != route.origin_as:
                attacker_as = candidate
                break

        new_path            = list(route.as_path[:-1]) + [attacker_as]
        route.as_path       = new_path
        route.origin_as     = attacker_as    # the hijacker's claim
        route.is_attack     = True
        route.attack_type   = "origin_hijack"
        route.attacker_as   = attacker_as
        return route

    def _path_shortening(self, route: BGPRoute) -> Optional[BGPRoute]:
        """
        Remove 1–2 intermediate hops from the middle of the path to make it
        appear shorter.  Requires at least 4 hops for a meaningful shortening
        (so that origin and endpoints stay intact).
        """
        path = list(route.as_path)
        if len(path) < 4:
            return None   # path too short to shorten meaningfully

        # Middle hops: indices 1 .. len-2 (excluding first and last)
        mid_indices = list(range(1, len(path) - 1))
        n_remove    = self.rng.randint(1, min(2, len(mid_indices)))
        remove_idxs = sorted(self.rng.sample(mid_indices, n_remove), reverse=True)

        for idx in remove_idxs:
            path.pop(idx)

        route.as_path     = path
        route.is_attack   = True
        route.attack_type = "path_shortening"
        route.attacker_as = path[0]   # the AS presenting the shortened path
        return route

    def _path_fabrication(self, route: BGPRoute) -> BGPRoute:
        """
        Insert 1–2 fake (attacker-controlled) AS numbers into the middle of
        the path.  These inserted ASes are not actually adjacent in the
        topology, so cryptographic checks will fail at those positions.
        """
        path = list(route.as_path)

        # Choose 1 or 2 positions in the middle to insert at
        n_insert = self.rng.randint(1, 2)
        for _ in range(n_insert):
            fake_as = self.rng.choice(self._all_ases)
            # Insert in the middle (not at origin or immediate peer positions)
            insert_pos = self.rng.randint(1, max(1, len(path) - 2))
            path.insert(insert_pos, fake_as)

        route.as_path     = path
        route.is_attack   = True
        route.attack_type = "path_fabrication"
        route.attacker_as = path[0]
        return route

    # ── Statistics ─────────────────────────────────────────────────────────

    @staticmethod
    def traffic_stats(routes: List[BGPRoute]) -> Dict:
        """Return attack/legitimate breakdown of a mixed route list."""
        total    = len(routes)
        attacks  = [r for r in routes if r.is_attack]
        legit    = total - len(attacks)
        by_type: Dict[str, int] = {}
        for r in attacks:
            by_type[r.attack_type] = by_type.get(r.attack_type, 0) + 1
        return {
            "total":          total,
            "legitimate":     legit,
            "attacks":        len(attacks),
            "attack_rate":    round(len(attacks) / total, 4) if total else 0,
            "by_type":        by_type,
        }
