"""
Route Leak Attack Injector
==========================
Adds Type-1/2/3 route leaks to the synthetic BGP traffic stream.

Leak types
----------
- route_leak_type1_full_prefix
- route_leak_type2_lateral
- route_leak_type3_intentional
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence

from src.bgp_simulator import BGPRoute
from src.topology import ASTopology, CUSTOMER, PEER


class RouteLeakInjector:
    """Inject route-leak attacks into a base set of legitimate routes."""

    LEAK_TYPES = (
        "route_leak_type1_full_prefix",
        "route_leak_type2_lateral",
        "route_leak_type3_intentional",
    )

    def __init__(
        self,
        topology: ASTopology,
        leak_type_dist: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        self.topology = topology
        self.rng = random.Random(seed)
        self._all_ases = list(topology.graph.nodes())

        if leak_type_dist is None:
            leak_type_dist = {
                "route_leak_type1_full_prefix": 0.45,
                "route_leak_type2_lateral": 0.35,
                "route_leak_type3_intentional": 0.20,
            }
        self.leak_type_dist = leak_type_dist

    def inject_n_leaks(self, base_routes: Sequence[BGPRoute], n_leaks: int) -> List[BGPRoute]:
        """Generate exactly n_leaks attacks (best effort, may be slightly lower)."""
        if not base_routes or n_leaks <= 0:
            return []

        leak_types = list(self.leak_type_dist.keys())
        probs = [self.leak_type_dist[t] for t in leak_types]

        attacks: List[BGPRoute] = []
        for _ in range(n_leaks):
            base = copy.deepcopy(self.rng.choice(base_routes))
            leak_type = self.rng.choices(leak_types, weights=probs, k=1)[0]
            attack = self._inject(base, leak_type)
            if attack is not None:
                attacks.append(attack)
        return attacks

    @staticmethod
    def leak_stats(routes: Sequence[BGPRoute]) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for r in routes:
            if not r.is_attack or not r.attack_type:
                continue
            if not r.attack_type.startswith("route_leak_type"):
                continue
            stats[r.attack_type] = stats.get(r.attack_type, 0) + 1
        return stats

    # ---- Leak constructors ---------------------------------------------

    def _inject(self, route: BGPRoute, leak_type: str) -> Optional[BGPRoute]:
        if leak_type == "route_leak_type1_full_prefix":
            return self._type1_full_prefix(route)
        if leak_type == "route_leak_type2_lateral":
            return self._type2_lateral(route)
        if leak_type == "route_leak_type3_intentional":
            return self._type3_intentional(route)
        return None

    def _type1_full_prefix(self, route: BGPRoute) -> Optional[BGPRoute]:
        """
        Customer leaks a provider-learned route to another provider.
        """
        attacker = self._pick_as_with_provider()
        if attacker is None:
            return None

        target_provider = self._pick_provider_of(attacker, exclude={route.as_path[0] if route.as_path else -1})
        if target_provider is None:
            return None

        path_from_attacker = self.topology.get_path(attacker, route.origin_as)
        if not path_from_attacker or len(path_from_attacker) < 2:
            return None

        leaked_path = [target_provider] + path_from_attacker
        leaked_path = self._dedupe_adjacent(leaked_path)

        route.as_path = leaked_path
        route.is_attack = True
        route.attack_type = "route_leak_type1_full_prefix"
        route.attacker_as = attacker
        setattr(route, "otc_present", True)
        setattr(route, "leak_subtype", "type1")
        return route

    def _type2_lateral(self, route: BGPRoute) -> Optional[BGPRoute]:
        """
        Peer leaks peer-learned routes to a provider.
        """
        attacker = self._pick_as_with_provider_and_peer()
        if attacker is None:
            return None

        provider = self._pick_provider_of(attacker)
        peer = self._pick_peer_of(attacker)
        if provider is None or peer is None:
            return None

        tail = self.topology.get_path(peer, route.origin_as)
        if not tail or len(tail) < 2:
            return None

        leaked_path = [provider, attacker, peer] + tail[1:]
        leaked_path = self._dedupe_adjacent(leaked_path)

        route.as_path = leaked_path
        route.is_attack = True
        route.attack_type = "route_leak_type2_lateral"
        route.attacker_as = attacker
        setattr(route, "otc_present", True)
        setattr(route, "leak_subtype", "type2")
        return route

    def _type3_intentional(self, route: BGPRoute) -> Optional[BGPRoute]:
        """
        Intentional strategic leak to attract traffic via a short-looking path.
        """
        attacker = self._pick_as_with_provider()
        if attacker is None:
            return None

        provider = self._pick_provider_of(attacker)
        if provider is None:
            return None

        tail = self.topology.get_path(attacker, route.origin_as)
        if not tail or len(tail) < 2:
            return None

        # Make the leaked path more attractive by removing middle hops.
        shortened = list(tail)
        if len(shortened) > 4:
            removable = list(range(1, len(shortened) - 1))
            n_remove = min(2, len(removable))
            remove_idx = sorted(self.rng.sample(removable, n_remove), reverse=True)
            for idx in remove_idx:
                shortened.pop(idx)

        leaked_path = [provider] + shortened
        leaked_path = self._dedupe_adjacent(leaked_path)

        route.as_path = leaked_path
        route.is_attack = True
        route.attack_type = "route_leak_type3_intentional"
        route.attacker_as = attacker
        setattr(route, "otc_present", True)
        setattr(route, "leak_subtype", "type3")
        return route

    # ---- Topology helpers ----------------------------------------------

    def _pick_as_with_provider(self) -> Optional[int]:
        candidates: List[int] = []
        for asn in self._all_ases:
            if self._providers_of(asn):
                candidates.append(asn)
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _pick_as_with_provider_and_peer(self) -> Optional[int]:
        candidates: List[int] = []
        for asn in self._all_ases:
            if self._providers_of(asn) and self._peers_of(asn):
                candidates.append(asn)
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _providers_of(self, asn: int) -> List[int]:
        out: List[int] = []
        for nbr in self.topology.graph.neighbors(asn):
            rel = self.topology.relationship(asn, nbr)
            if rel is not None and rel < 0:
                out.append(nbr)
        return out

    def _peers_of(self, asn: int) -> List[int]:
        out: List[int] = []
        for nbr in self.topology.graph.neighbors(asn):
            rel = self.topology.relationship(asn, nbr)
            if rel == PEER:
                out.append(nbr)
        return out

    def _pick_provider_of(self, asn: int, exclude: Optional[set] = None) -> Optional[int]:
        exclude = exclude or set()
        providers = [p for p in self._providers_of(asn) if p not in exclude]
        if not providers:
            return None
        return self.rng.choice(providers)

    def _pick_peer_of(self, asn: int) -> Optional[int]:
        peers = self._peers_of(asn)
        if not peers:
            return None
        return self.rng.choice(peers)

    @staticmethod
    def _dedupe_adjacent(path: List[int]) -> List[int]:
        if not path:
            return path
        out = [path[0]]
        for asn in path[1:]:
            if asn != out[-1]:
                out.append(asn)
        return out
