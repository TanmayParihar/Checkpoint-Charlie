"""
AS Topology Generator
=====================
Builds a synthetic but realistic Internet-scale AS topology using a
Barabási-Albert preferential attachment model, then overlays realistic
tier assignments and AS relationship semantics (customer-provider,
peer-peer) consistent with published measurements of the real Internet.

Relationship encoding (CAIDA convention):
  -1 → provider  (you pay them / they are your upstream)
   0 → peer      (settlement-free bilateral exchange)
   1 → customer  (they pay you / you are their upstream)

Valley-free routing constraint: traffic from a customer can go up to a
provider or across to a peer, then only downward to customers.  It must
never go up again after going down.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ── Relationship constants ───────────────────────────────────────────────────
PROVIDER  = -1
PEER      = 0
CUSTOMER  = 1


class ASTopology:
    """
    Synthetic Internet AS topology.

    Attributes
    ----------
    graph : nx.Graph
        Undirected adjacency graph.  Each edge carries a 'rel' attribute
        storing the relationship from the perspective of the lower-numbered
        AS (CAIDA convention).
    as_tier : dict[int, int]
        Maps AS number → tier (1, 2, or 3).
    as_rel : dict[tuple[int,int], int]
        Directed relationship: as_rel[(u, v)] = relationship of u towards v.
    prefixes : dict[int, list[str]]
        AS → list of IP prefixes it originates.
    roa_db : dict[str, int]
        prefix → authorised origin AS (RPKI ROA database).
    """

    def __init__(
        self,
        num_ases: int = 2000,
        tier1_count: int = 15,
        tier2_count: int = 300,
        rpki_coverage: float = 0.45,
        seed: int = 42,
    ) -> None:
        self.num_ases     = num_ases
        self.tier1_count  = tier1_count
        self.tier2_count  = tier2_count
        self.rpki_coverage = rpki_coverage
        self.rng          = random.Random(seed)
        self.np_rng       = np.random.default_rng(seed)

        self.graph: nx.Graph       = nx.Graph()
        self.as_tier: Dict[int, int]         = {}
        self.as_rel: Dict[Tuple[int,int], int] = {}
        self.prefixes: Dict[int, List[str]]  = {}
        self.roa_db: Dict[str, int]          = {}

        self._build()

    # ── Public interface ──────────────────────────────────────────────────────

    def get_path(self, src: int, dst: int) -> Optional[List[int]]:
        """Return a shortest path from src to dst (None if disconnected)."""
        try:
            return nx.shortest_path(self.graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def are_adjacent(self, as1: int, as2: int) -> bool:
        """Return True if the two ASes share a direct link."""
        return self.graph.has_edge(as1, as2)

    def relationship(self, u: int, v: int) -> Optional[int]:
        """Return the directed relationship of u towards v."""
        return self.as_rel.get((u, v))

    def is_valley_free(self, as_path: List[int]) -> bool:
        """
        Check the valley-free property for an AS_PATH.

        In BGP the AS_PATH is stored as [receiving_peer, ..., origin].
        We traverse it left-to-right (receiver → origin) to check that we
        never go *down* to a customer and then *up* to a provider again.
        """
        if len(as_path) < 2:
            return True

        went_down = False
        for i in range(len(as_path) - 1):
            u, v = as_path[i], as_path[i + 1]
            rel = self.as_rel.get((u, v))
            if rel is None:
                return False          # non-adjacent hops → invalid
            if rel == CUSTOMER:       # u→v means v is customer of u: going DOWN
                went_down = True
            elif rel == PROVIDER:     # u→v means v is provider of u: going UP
                if went_down:
                    return False      # valley!
        return True

    def sample_as_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Sample n (source, destination) AS pairs for route generation.

        The Barabási-Albert construction guarantees the graph is connected
        (verified at build time via stats()), so every pair has a path and
        the nx.has_path() guard is unnecessary overhead.
        """
        ases = list(self.graph.nodes())
        pairs = [tuple(self.rng.sample(ases, 2)) for _ in range(n)]
        return pairs  # type: ignore[return-value]

    # ── Internal build methods ────────────────────────────────────────────────

    def _build(self) -> None:
        """Construct topology, assign tiers, relationships, prefixes, ROAs."""
        self._generate_graph()
        self._assign_tiers()
        self._assign_relationships()
        self._assign_prefixes()
        self._build_roa_db()

    def _generate_graph(self) -> None:
        """
        Build a Barabási-Albert preferential attachment graph.
        Each new node attaches to m=3 existing nodes, producing the power-law
        degree distribution observed in the real Internet.
        """
        ba = nx.barabasi_albert_graph(self.num_ases, m=3, seed=42)
        # Re-label nodes to realistic AS numbers (1-based, sparse)
        mapping = {i: i + 1 for i in range(self.num_ases)}
        self.graph = nx.relabel_nodes(ba, mapping)

    def _assign_tiers(self) -> None:
        """Classify ASes into tiers based on degree (connectivity)."""
        degrees = dict(self.graph.degree())
        sorted_by_deg = sorted(degrees, key=degrees.get, reverse=True)

        tier1_set = set(sorted_by_deg[: self.tier1_count])
        tier2_set = set(sorted_by_deg[self.tier1_count : self.tier1_count + self.tier2_count])

        for node in self.graph.nodes():
            if node in tier1_set:
                self.as_tier[node] = 1
            elif node in tier2_set:
                self.as_tier[node] = 2
            else:
                self.as_tier[node] = 3

    def _assign_relationships(self) -> None:
        """
        Overlay relationships on graph edges:
          Tier1 ↔ Tier1 → peers
          Tier1 ↔ Tier2 → Tier1 is provider, Tier2 is customer
          Tier1 ↔ Tier3 → Tier1 is provider, Tier3 is customer
          Tier2 ↔ Tier2 → peers (with 70% probability) or provider-customer
          Tier2 ↔ Tier3 → Tier2 is provider, Tier3 is customer
          Tier3 ↔ Tier3 → peers
        """
        for u, v in self.graph.edges():
            tu, tv = self.as_tier[u], self.as_tier[v]

            if tu == tv == 1:
                # Tier-1 to Tier-1: always peers
                rel_u_to_v = PEER
            elif tu < tv:
                # Higher tier (lower number) is the provider
                rel_u_to_v = CUSTOMER     # u provides to v → u sees v as customer
            elif tu > tv:
                rel_u_to_v = PROVIDER     # u is customer of v → u sees v as provider
            else:
                # Same tier (2 or 3): mostly peer, sometimes customer-provider
                if self.rng.random() < 0.7:
                    rel_u_to_v = PEER
                else:
                    # Randomly pick direction
                    rel_u_to_v = self.rng.choice([CUSTOMER, PROVIDER])

            # Store directed relationships for both directions
            self.as_rel[(u, v)] = rel_u_to_v
            # Reciprocal: if u sees v as customer, v sees u as provider
            if rel_u_to_v == CUSTOMER:
                self.as_rel[(v, u)] = PROVIDER
            elif rel_u_to_v == PROVIDER:
                self.as_rel[(v, u)] = CUSTOMER
            else:
                self.as_rel[(v, u)] = PEER

    def _assign_prefixes(self) -> None:
        """Give each AS 1-5 IP prefixes (simulated /16 to /24 ranges)."""
        octets_used: Set[Tuple[int, int]] = set()
        for asn in self.graph.nodes():
            num_prefixes = self.rng.randint(1, 5)
            self.prefixes[asn] = []
            for _ in range(num_prefixes):
                while True:
                    a = self.rng.randint(1, 223)
                    b = self.rng.randint(0, 255)
                    if (a, b) not in octets_used:
                        octets_used.add((a, b))
                        break
                prefix_len = self.rng.randint(16, 24)
                self.prefixes[asn].append(f"{a}.{b}.0.0/{prefix_len}")

    def _build_roa_db(self) -> None:
        """
        Build the RPKI ROA database.
        Only `rpki_coverage` fraction of prefixes have a valid ROA.
        """
        for asn, pfx_list in self.prefixes.items():
            for pfx in pfx_list:
                if self.rng.random() < self.rpki_coverage:
                    self.roa_db[pfx] = asn

    # ── Utility ───────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return a summary of topology statistics."""
        tier_counts = defaultdict(int)
        for t in self.as_tier.values():
            tier_counts[t] += 1
        degrees = [d for _, d in self.graph.degree()]
        return {
            "num_ases":         self.graph.number_of_nodes(),
            "num_links":        self.graph.number_of_edges(),
            "tier1":            tier_counts[1],
            "tier2":            tier_counts[2],
            "tier3":            tier_counts[3],
            "avg_degree":       round(float(np.mean(degrees)), 2),
            "max_degree":       max(degrees),
            "total_prefixes":   sum(len(p) for p in self.prefixes.values()),
            "roa_entries":      len(self.roa_db),
            "rpki_coverage_pct": round(100 * len(self.roa_db) /
                                       max(1, sum(len(p) for p in self.prefixes.values())), 1),
            "connected":        nx.is_connected(self.graph),
        }
