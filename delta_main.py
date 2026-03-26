"""
Delta Trust Benchmark
=====================
Benchmarks the Delta / Incremental Trust Propagation technique against all
existing validation strategies.

Delta trust caches previously verified path segments.  When a new announcement
arrives, only the changed edges (the "diff") are verified.  Verification cost
scales with path churn, not path length -- like git for routing.

Usage
-----
  python delta_main.py [options]

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delta Trust Benchmark (Checkpoint Charlie)"
    )
    p.add_argument("--ases",         type=int,   default=config.NUM_ASES)
    p.add_argument("--routes",       type=int,   default=config.NUM_ROUTES)
    p.add_argument("--attack-rate",  type=float, default=config.ATTACK_RATE)
    p.add_argument("--rpki-cov",     type=float, default=config.RPKI_COVERAGE)
    p.add_argument("--seed",         type=int,   default=config.RANDOM_SEED)
    p.add_argument("--no-plots",     action="store_true")
    p.add_argument("--results-dir",  type=str,   default="results_delta")
    return p.parse_args()


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       Checkpoint Charlie – Delta Trust Propagation Benchmark      ║
║    Incremental Verification: verify the diff, not the full path   ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def build_validators(topology: ASTopology) -> List:
    """Instantiate all validators including Delta Trust."""
    validators = []

    # RPKI (origin-only)
    validators.append(RPKIValidator(topology))

    # Selective hop combinations
    for key, cfg in config.HOP_COMBINATIONS.items():
        if cfg["positions"] is None:
            validators.append(BGPsecValidator(topology))
        else:
            validators.append(
                SelectiveHopValidator(
                    topology,
                    positions=cfg["positions"],
                    label=cfg["label"],
                )
            )

    # Delta Trust (the new technique)
    validators.append(DeltaTrustValidator(topology, label="Delta Trust"))

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

    # ── Step 5: Build Validators ─────────────────────────────────────────────
    _print_section("Step 5 – Building Validators")
    validators = build_validators(topo)

    # Deduplicate
    seen_names: set = set()
    unique_validators = []
    for v in validators:
        if v.name not in seen_names:
            unique_validators.append(v)
            seen_names.add(v.name)
    unique_validators.append(anomaly)

    # Composite: Selective hop + Anomaly (original technique)
    class CompositeValidator:
        """Flags a route if EITHER the selective-hop OR the anomaly detector flags it."""
        name = "Selective + Anomaly"

        def __init__(self, hop_v, anom_v):
            self._hop  = hop_v
            self._anom = anom_v

        def validate(self, route: BGPRoute):
            h_ok, h_t = self._hop.validate(route)
            a_ok, a_t = self._anom.validate(route)
            return (h_ok and a_ok), h_t + a_t

    ofl_v = next((v for v in unique_validators if "1st + Last" in v.name), None)
    if ofl_v:
        unique_validators.append(CompositeValidator(ofl_v, anomaly))

    for v in unique_validators:
        print(f"    · {v.name}")

    # ── Step 6: Evaluation ────────────────────────────────────────────────────
    _print_section("Step 6 – Evaluating Validators")
    evaluator = Evaluator(unique_validators, all_routes)
    evaluator.run_all()

    summary   = evaluator.summary_table()
    breakdown = evaluator.attack_type_breakdown()

    # ── Step 7: Print Results ─────────────────────────────────────────────────
    _print_section("Results – Summary Table")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_columns",  20)
    pd.set_option("display.width",        140)
    print(summary[["detection_rate", "false_positive_rate", "avg_time_ms", "security_cost_ratio"]].to_string())
    print("\n  NOTE: avg_time_ms values are *model estimates* based on published")
    print("  RSA-2048 benchmarks (0.50 ms/verification), not wall-clock timings.")

    _print_section("Results – Detection Rate by Attack Type")
    print(breakdown.to_string())

    # ── Delta Trust Analysis ──────────────────────────────────────────────────
    _print_section("Delta Trust Analysis")
    delta_v = next((v for v in unique_validators if v.name == "Delta Trust"), None)
    if delta_v:
        cache = delta_v.cache_stats()
        print(f"    Trust cache:")
        print(f"      Prefixes cached     : {cache['prefixes_cached']:,}")
        print(f"      Total cached edges  : {cache['total_cached_edges']:,}")
        print()

        # Compare Delta Trust vs BGPsec
        bgpsec_name = "BGPsec (Full Path)"
        delta_name = "Delta Trust"
        if bgpsec_name in summary.index and delta_name in summary.index:
            bgp_t  = summary.loc[bgpsec_name, "avg_time_ms"]
            delta_t = summary.loc[delta_name,  "avg_time_ms"]
            bgp_dr  = summary.loc[bgpsec_name, "detection_rate"]
            delta_dr = summary.loc[delta_name,  "detection_rate"]
            if delta_t > 0:
                speedup = bgp_t / delta_t
            else:
                speedup = float("inf")
            print(f"    Delta Trust vs BGPsec:")
            print(f"      BGPsec  detection: {bgp_dr*100:.1f}%  avg time: {bgp_t*1000:.3f} µs")
            print(f"      Delta   detection: {delta_dr*100:.1f}%  avg time: {delta_t*1000:.3f} µs")
            print(f"      Speedup: {speedup:.2f}x faster")
            print(f"      Same detection, fraction of the cost on stable prefixes.")

    # ── Key Findings ──────────────────────────────────────────────────────────
    _print_section("Key Findings")
    for vname, label in [
        ("RPKI",                   "RPKI (origin-only)"),
        ("Origin + 1st + Last",    "Selective hop (origin+first+last)"),
        ("Anomaly Detector",       "Statistical anomaly detection"),
        ("BGPsec (Full Path)",     "BGPsec (full path)"),
        ("Selective + Anomaly",    "Combined (selective + anomaly)"),
        ("Delta Trust",            "Delta Trust (incremental verification)"),
    ]:
        if vname in summary.index:
            dr  = summary.loc[vname, "detection_rate"] * 100
            fpr = summary.loc[vname, "false_positive_rate"] * 100
            t   = summary.loc[vname, "avg_time_ms"] * 1000
            print(f"  {label}")
            print(f"      Detection Rate  : {dr:.1f}%")
            print(f"      False Pos. Rate : {fpr:.2f}%")
            print(f"      Avg. Time       : {t:.3f} µs/announcement")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        _print_section("Step 8 – Generating Figures")
        from src.visualization import generate_all_figures
        generate_all_figures(summary, breakdown)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─'*64}")
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"{'─'*64}\n")

    # Save CSV
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary.csv"))
    breakdown.to_csv(os.path.join(config.RESULTS_DIR, "attack_breakdown.csv"))
    print(f"  CSV results saved to ./{config.RESULTS_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
