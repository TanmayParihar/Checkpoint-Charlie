"""
Delta + Original Combined Benchmark
====================================
Combines the Delta / Incremental Trust Propagation technique with the original
Selective Hop Verification + Statistical Anomaly Detection approach, and
benchmarks every strategy side-by-side:

  1. All original validators (RPKI, Selective Hop variants, BGPsec, Anomaly)
  2. Original combined  : Selective Hop + Anomaly
  3. Delta Trust alone  : incremental edge verification
  4. Delta + Original   : Delta Trust + Selective Hop + Anomaly (triple combo)

The triple combination leverages three complementary strengths:
  - Delta Trust        → cheap verification of unchanged path segments (cost)
  - Selective Hop      → targeted cryptographic checks at critical positions
  - Anomaly Detection  → statistical detection of subtle path manipulation

A route is flagged if ANY of the three methods flags it.

Usage
-----
  python delta_combined_main.py [options]

Options are the same as main.py (--ases, --routes, --attack-rate, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import pandas as pd

import config
from src.attack_injector import AttackInjector
from src.bgp_simulator import BGPRoute, BGPSimulator
from src.evaluator import Evaluator
from src.topology import ASTopology
from src.validators.anomaly import AnomalyDetector
from src.validators.bgpsec import BGPsecValidator
from src.validators.delta_trust import DeltaTrustValidator
from src.validators.rpki import RPKIValidator
from src.validators.selective_hop import SelectiveHopValidator


# ── Composite validators ─────────────────────────────────────────────────────

class SelectiveAnomalyComposite:
    """Original combined: flags if EITHER selective-hop OR anomaly flags it."""
    name = "Selective + Anomaly"

    def __init__(self, hop_v, anom_v):
        self._hop  = hop_v
        self._anom = anom_v

    def validate(self, route: BGPRoute):
        h_ok, h_t = self._hop.validate(route)
        a_ok, a_t = self._anom.validate(route)
        return (h_ok and a_ok), h_t + a_t


class DeltaSelectiveAnomalyComposite:
    """
    Triple combination: Delta Trust + Selective Hop + Anomaly Detection.

    Flags a route if ANY of the three methods flags it.

    Cost model: runs delta trust first (cheapest on cache hits), then selective
    hop on the critical positions, then the anomaly detector.  Total time is
    the sum of all three (worst case), but detection is maximised by the union
    of all three signal sources.
    """
    name = "Delta + Selective + Anomaly"

    def __init__(self, delta_v, hop_v, anom_v):
        self._delta = delta_v
        self._hop   = hop_v
        self._anom  = anom_v

    def validate(self, route: BGPRoute):
        d_ok, d_t = self._delta.validate(route)
        h_ok, h_t = self._hop.validate(route)
        a_ok, a_t = self._anom.validate(route)
        total_t = d_t + h_t + a_t
        # Reject if ANY method flags the route
        is_valid = d_ok and h_ok and a_ok
        return is_valid, total_t


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delta + Original Combined Benchmark (Checkpoint Charlie)"
    )
    p.add_argument("--ases",         type=int,   default=config.NUM_ASES)
    p.add_argument("--routes",       type=int,   default=config.NUM_ROUTES)
    p.add_argument("--attack-rate",  type=float, default=config.ATTACK_RATE)
    p.add_argument("--rpki-cov",     type=float, default=config.RPKI_COVERAGE)
    p.add_argument("--seed",         type=int,   default=config.RANDOM_SEED)
    p.add_argument("--no-plots",     action="store_true")
    p.add_argument("--results-dir",  type=str,   default="results_delta_combined")
    return p.parse_args()


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║     Checkpoint Charlie – Delta + Original Combined Benchmark             ║
║  Delta Trust  ×  Selective Hop Verification  ×  Statistical Anomaly       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _print_section(title: str) -> None:
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")


def build_validators(topology: ASTopology, anomaly: AnomalyDetector) -> List:
    """Instantiate ALL validators: originals + delta + combined variants."""
    validators = []

    # ── Original validators ──────────────────────────────────────────────────
    # RPKI
    validators.append(RPKIValidator(topology))

    # Selective hop combinations
    ofl_v = None  # will hold the origin+first+last validator
    for key, cfg in config.HOP_COMBINATIONS.items():
        if cfg["positions"] is None:
            validators.append(BGPsecValidator(topology))
        else:
            v = SelectiveHopValidator(
                topology,
                positions=cfg["positions"],
                label=cfg["label"],
            )
            validators.append(v)
            if "1st + Last" in cfg["label"]:
                ofl_v = v

    # Anomaly detector
    validators.append(anomaly)

    # Original composite: Selective + Anomaly
    if ofl_v:
        validators.append(SelectiveAnomalyComposite(ofl_v, anomaly))

    # ── Delta Trust (standalone) ─────────────────────────────────────────────
    delta_v = DeltaTrustValidator(topology, label="Delta Trust")
    validators.append(delta_v)

    # ── Triple combination: Delta + Selective + Anomaly ──────────────────────
    if ofl_v:
        # Use a separate delta instance so the cache is independent
        delta_v_combined = DeltaTrustValidator(
            topology, label="Delta+Sel+Anom (internal)"
        )
        validators.append(
            DeltaSelectiveAnomalyComposite(delta_v_combined, ofl_v, anomaly)
        )

    return validators


# ── Main simulation ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print(BANNER)
    t_start = time.perf_counter()

    config.RESULTS_DIR = args.results_dir

    # ── Step 1: Topology ─────────────────────────────────────────────────────
    _print_section("Step 1 – Building AS Topology")
    topo = ASTopology(
        num_ases      = args.ases,
        tier1_count   = config.TIER1_COUNT,
        tier2_count   = config.TIER2_COUNT,
        rpki_coverage = args.rpki_cov,
        seed          = args.seed,
    )
    stats = topo.stats()
    for k, v in stats.items():
        print(f"    {k:<25} {v}")

    # ── Step 2: Normal Routes ────────────────────────────────────────────────
    _print_section("Step 2 – Generating Normal BGP Routes")
    n_normal = int(args.routes * (1 - args.attack_rate))
    sim = BGPSimulator(topo, num_routes=n_normal, seed=args.seed)
    normal_routes = sim.generate_normal_routes(n_normal)
    plen_stats = sim.path_length_stats(normal_routes)
    print(f"    Generated {len(normal_routes):,} normal routes")
    print(f"    Path length: mean={plen_stats['mean']:.2f}, "
          f"std={plen_stats['std']:.2f}, "
          f"min={plen_stats['min']}, max={plen_stats['max']}")

    # ── Step 3: Attack Injection ─────────────────────────────────────────────
    _print_section("Step 3 – Injecting Attack Routes")
    injector = AttackInjector(
        topology         = topo,
        attack_rate      = args.attack_rate,
        attack_type_dist = config.ATTACK_TYPE_DIST,
        seed             = args.seed,
    )
    all_routes = injector.mix_attacks(normal_routes)
    traffic_stats = AttackInjector.traffic_stats(all_routes)
    print(f"    Total announcements : {traffic_stats['total']:,}")
    print(f"    Legitimate          : {traffic_stats['legitimate']:,}")
    print(f"    Attacks             : {traffic_stats['attacks']:,} ({traffic_stats['attack_rate']*100:.1f}%)")
    for atype, cnt in traffic_stats["by_type"].items():
        print(f"      · {atype:<25} {cnt:,}")

    # ── Step 4: Anomaly Detector Training ────────────────────────────────────
    _print_section("Step 4 – Training Anomaly Detector")
    anomaly = AnomalyDetector(topo, threshold=config.ANOMALY_SCORE_THRESHOLD)
    anomaly.train(normal_routes)
    print(f"    Trained on {len(normal_routes):,} routes")
    print(f"    Known ASes          : {len(anomaly._known_ases):,}")
    print(f"    Tracked prefixes    : {len(anomaly._prefix_origins):,}")

    # ── Step 5: Build ALL Validators ─────────────────────────────────────────
    _print_section("Step 5 – Building All Validators")
    validators = build_validators(topo, anomaly)

    # Deduplicate by name
    seen_names: set = set()
    unique_validators = []
    for v in validators:
        if v.name not in seen_names:
            unique_validators.append(v)
            seen_names.add(v.name)

    for v in unique_validators:
        print(f"    · {v.name}")

    # ── Step 6: Evaluation ────────────────────────────────────────────────────
    _print_section("Step 6 – Evaluating All Validators")
    evaluator = Evaluator(unique_validators, all_routes)
    evaluator.run_all()

    summary   = evaluator.summary_table()
    breakdown = evaluator.attack_type_breakdown()

    # ── Step 7: Print Results ─────────────────────────────────────────────────
    _print_section("Results – Full Summary Table")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_columns",  20)
    pd.set_option("display.width",        140)
    print(summary[["detection_rate", "false_positive_rate", "avg_time_ms", "security_cost_ratio"]].to_string())
    print("\n  NOTE: avg_time_ms values are *model estimates* based on published")
    print("  RSA-2048 benchmarks (0.50 ms/verification), not wall-clock timings.")

    _print_section("Results – Detection Rate by Attack Type")
    print(breakdown.to_string())

    # ── Comparative Analysis ──────────────────────────────────────────────────
    _print_section("Comparative Analysis – All Techniques")

    technique_groups = [
        ("Baseline",       ["RPKI"]),
        ("Selective Hop",  ["Selective: Origin Only", "Origin + 1st Hop",
                            "Origin + Last Hop", "Origin + 1st + Last",
                            "Origin + 2 Hops + Last"]),
        ("Full Path",      ["BGPsec (Full Path)"]),
        ("Statistical",    ["Anomaly Detector"]),
        ("Original Combined", ["Selective + Anomaly"]),
        ("Delta Trust",    ["Delta Trust"]),
        ("Delta + Original", ["Delta + Selective + Anomaly"]),
    ]

    for group_name, vnames in technique_groups:
        present = [n for n in vnames if n in summary.index]
        if not present:
            continue
        print(f"\n  ── {group_name} ──")
        for vname in present:
            dr  = summary.loc[vname, "detection_rate"] * 100
            fpr = summary.loc[vname, "false_positive_rate"] * 100
            t   = summary.loc[vname, "avg_time_ms"] * 1000
            scr = summary.loc[vname, "security_cost_ratio"]
            print(f"    {vname}")
            print(f"        Detection: {dr:.1f}%  |  FPR: {fpr:.2f}%  |  "
                  f"Time: {t:.3f} µs  |  Sec/Cost: {scr:.2f}")

    # ── Delta Trust Cache Stats ──────────────────────────────────────────────
    _print_section("Delta Trust Cache Analysis")
    for v in unique_validators:
        if hasattr(v, 'cache_stats'):
            cache = v.cache_stats()
            print(f"    {v.name}:")
            print(f"      Prefixes cached     : {cache['prefixes_cached']:,}")
            print(f"      Total cached edges  : {cache['total_cached_edges']:,}")
        elif hasattr(v, '_delta') and hasattr(v._delta, 'cache_stats'):
            cache = v._delta.cache_stats()
            print(f"    {v.name} (delta component):")
            print(f"      Prefixes cached     : {cache['prefixes_cached']:,}")
            print(f"      Total cached edges  : {cache['total_cached_edges']:,}")

    # ── Head-to-Head Comparison ──────────────────────────────────────────────
    _print_section("Head-to-Head: Original vs Delta vs Combined")

    comparisons = [
        ("Selective + Anomaly",          "Original (Sel+Anom)"),
        ("Delta Trust",                  "Delta Trust alone"),
        ("Delta + Selective + Anomaly",  "Delta + Original"),
        ("BGPsec (Full Path)",           "BGPsec (baseline)"),
    ]

    for vname, label in comparisons:
        if vname in summary.index:
            dr  = summary.loc[vname, "detection_rate"] * 100
            fpr = summary.loc[vname, "false_positive_rate"] * 100
            t   = summary.loc[vname, "avg_time_ms"] * 1000
            scr = summary.loc[vname, "security_cost_ratio"]
            print(f"  {label:<28}  DR: {dr:5.1f}%  FPR: {fpr:5.2f}%  "
                  f"Time: {t:8.3f} µs  Eff: {scr:8.2f}")

    # Show improvement of Delta+Original over Original
    orig_name = "Selective + Anomaly"
    combo_name = "Delta + Selective + Anomaly"
    if orig_name in summary.index and combo_name in summary.index:
        orig_dr  = summary.loc[orig_name,  "detection_rate"]
        combo_dr = summary.loc[combo_name, "detection_rate"]
        orig_fpr  = summary.loc[orig_name,  "false_positive_rate"]
        combo_fpr = summary.loc[combo_name, "false_positive_rate"]
        dr_delta = (combo_dr - orig_dr) * 100
        print(f"\n  Detection improvement (Delta+Original vs Original): "
              f"{dr_delta:+.2f} percentage points")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        _print_section("Step 8 – Generating Figures")
        from src.visualization import generate_all_figures
        generate_all_figures(summary, breakdown)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─'*72}")
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"{'─'*72}\n")

    # Save CSV
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary.csv"))
    breakdown.to_csv(os.path.join(config.RESULTS_DIR, "attack_breakdown.csv"))
    print(f"  CSV results saved to ./{config.RESULTS_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
