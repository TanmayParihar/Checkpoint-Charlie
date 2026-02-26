"""
Data Acquisition Script
========================
Downloads publicly available BGP-related datasets to enrich the simulation.

Datasets targeted
-----------------
1. CAIDA AS Relationships (serial-1)
   URL  : https://publicdata.caida.org/datasets/as-relationships/serial-1/
   File : YYYYMMDD.as-rel.txt.bz2
   Use  : Build a real-topology-based AS graph instead of the synthetic BA model.

2. CAIDA AS Organizations
   URL  : https://publicdata.caida.org/datasets/as-organizations/
   Use  : Map ASNs to real organisation names for richer reporting.

Usage
-----
  python data/download_data.py [--date YYYYMMDD] [--out-dir data/]

If a download fails the script prints a warning but does NOT crash – the main
simulation falls back to the synthetic topology automatically.
"""

from __future__ import annotations

import argparse
import bz2
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ── CAIDA helpers ─────────────────────────────────────────────────────────────

CAIDA_AS_REL_BASE = "https://publicdata.caida.org/datasets/as-relationships/serial-1/"
CAIDA_AS_ORG_BASE = "https://publicdata.caida.org/datasets/as-organizations/"


def _latest_as_rel_date() -> str:
    """Return YYYYMMDD string for the most recent AS-rel snapshot."""
    today = date.today()
    # Files are typically published monthly; try the first of recent months
    for months_back in range(0, 6):
        d = date(today.year, today.month, 1) - timedelta(days=months_back * 30)
        yield d.strftime("%Y%m01")


def download_as_relationships(out_dir: str = "data", target_date: Optional[str] = None) -> Optional[str]:
    """
    Download the CAIDA AS-relationship file for the given date (or latest).

    Returns the path to the decompressed text file, or None on failure.
    """
    if not HAS_REQUESTS:
        print("[WARN] requests library not installed – skipping download.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    dates_to_try = [target_date] if target_date else list(_latest_as_rel_date())

    for d in dates_to_try:
        fname_bz2 = f"{d}.as-rel.txt.bz2"
        fname_txt = os.path.join(out_dir, f"{d}.as-rel.txt")

        if os.path.exists(fname_txt):
            print(f"[OK] Already downloaded: {fname_txt}")
            return fname_txt

        url = f"{CAIDA_AS_REL_BASE}{fname_bz2}"
        print(f"[INFO] Trying {url} …")
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] {exc}")
            continue

        # Decompress bz2 on-the-fly
        raw = bz2.decompress(resp.content)
        with open(fname_txt, "wb") as fh:
            fh.write(raw)
        print(f"[OK] Saved: {fname_txt}  ({len(raw)//1024} KB)")
        return fname_txt

    print("[WARN] Could not download CAIDA AS-relationship file.")
    return None


def parse_as_relationships(filepath: str) -> List[Tuple[int, int, int]]:
    """
    Parse a CAIDA AS-relationship text file.

    Format::
        # comment lines
        <as1>|<as2>|<rel>|<source>
        rel: -1 = provider of as2, 0 = peer

    Returns list of (as1, as2, relationship) tuples.
    """
    edges: List[Tuple[int, int, int]] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            try:
                as1 = int(parts[0])
                as2 = int(parts[1])
                rel = int(parts[2])
                edges.append((as1, as2, rel))
            except ValueError:
                continue
    return edges


def load_topology_from_caida(filepath: str):
    """
    Build a networkx graph from a CAIDA AS-relationship file.
    Returns (graph, as_rel_dict) compatible with ASTopology internals,
    or (None, None) if networkx is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] networkx not installed – cannot build CAIDA topology.")
        return None, None

    edges = parse_as_relationships(filepath)
    print(f"[INFO] Parsed {len(edges):,} AS relationships from {filepath}")

    G = nx.Graph()
    as_rel: Dict[Tuple[int, int], int] = {}

    for as1, as2, rel in edges:
        G.add_edge(as1, as2)
        # CAIDA rel=-1 means as1 is provider of as2
        # We store as as_rel[(as1,as2)] = CUSTOMER (as2 pays as1)
        #                as_rel[(as2,as1)] = PROVIDER
        if rel == -1:
            as_rel[(as1, as2)] = 1    # as1 → as2: as2 is customer
            as_rel[(as2, as1)] = -1   # as2 → as1: as1 is provider
        elif rel == 0:
            as_rel[(as1, as2)] = 0
            as_rel[(as2, as1)] = 0
        else:
            as_rel[(as1, as2)] = rel
            as_rel[(as2, as1)] = -rel if rel != 0 else 0

    print(f"[INFO] CAIDA graph: {G.number_of_nodes():,} ASes, {G.number_of_edges():,} links")
    return G, as_rel


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Download BGP datasets for Checkpoint Charlie")
    p.add_argument("--date",    type=str, default=None,   help="YYYYMMDD of CAIDA snapshot (default: latest)")
    p.add_argument("--out-dir", type=str, default="data", help="Directory to save downloaded files")
    args = p.parse_args()

    print("=" * 60)
    print("  Checkpoint Charlie – Dataset Downloader")
    print("=" * 60)

    filepath = download_as_relationships(out_dir=args.out_dir, target_date=args.date)

    if filepath:
        edges = parse_as_relationships(filepath)
        print(f"\nSummary:")
        print(f"  Relationships loaded : {len(edges):,}")
        unique_ases = len({a for e in edges for a in e[:2]})
        print(f"  Unique ASes          : {unique_ases:,}")
        print(f"\n[OK] Data ready at: {filepath}")
        print("     Pass this file to ASTopology.load_from_caida() (future feature).")
    else:
        print("\n[INFO] Falling back to synthetic topology in the main simulation.")
        print("       Run:  python main.py")


if __name__ == "__main__":
    main()
