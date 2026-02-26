"""
Checkpoint Charlie – BGP Routing Security Simulation
=====================================================
Main entry point for the simulation.

Usage
-----
  python main.py [options]

Options
-------
  --ases N          Number of ASes to simulate    (default: 2000)
  --routes N        Total route announcements      (default: 20000)
  --attack-rate F   Fraction that are attacks      (default: 0.15)
  --rpki-cov F      RPKI ROA coverage fraction     (default: 0.45)
  --seed N          Random seed                    (default: 42)
  --no-plots        Skip generating figures
  --results-dir D   Output directory               (default: results)

Pipeline
--------
1. Build AS topology (Barabási-Albert + tier/relationship assignment)
2. Generate normal BGP route stream
3. Inject attacks (origin hijack, path shortening, path fabrication)
4. Train anomaly detector on the normal-route history
5. Build all validators
6. Evaluate all validators against the mixed traffic
7. Print summary tables
8. Save figures to results/
"""

from __future__ import annotations

import argparse
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
from src.validators.rpki import RPKIValidator
from src.validators.selective_hop import SelectiveHopValidator


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BGP Routing Security Simulation (Checkpoint Charlie)"
    )
    p.add_argument("--ases",         type=int,   default=config.NUM_ASES,       help="Number of simulated ASes")
    p.add_argument("--routes",       type=int,   default=config.NUM_ROUTES,     help="Total route announcements")
    p.add_argument("--attack-rate",  type=float, default=config.ATTACK_RATE,    help="Fraction of traffic that is attacks")
    p.add_argument("--rpki-cov",     type=float, default=config.RPKI_COVERAGE,  help="RPKI ROA coverage fraction")
    p.add_argument("--seed",         type=int,   default=config.RANDOM_SEED,    help="Random seed")
    p.add_argument("--no-plots",     action="store_true",                        help="Skip generating figures")
    p.add_argument("--results-dir",  type=str,   default=config.RESULTS_DIR,    help="Output directory for figures")
    return p.parse_args()


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          Checkpoint Charlie – BGP Routing Security Sim           ║
║  Selective Hop Verification  ×  Statistical Anomaly Detection    ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def build_validators(topology: ASTopology) -> List:
    """Instantiate all validators for benchmarking."""
    validators = []

    # RPKI (origin-only)
    validators.append(RPKIValidator(topology))

    # Selective hop combinations
    for key, cfg in config.HOP_COMBINATIONS.items():
        if cfg["positions"] is None:
            # Full path → BGPsec
            validators.append(BGPsecValidator(topology))
        else:
            validators.append(
                SelectiveHopValidator(
                    topology,
                    positions=cfg["positions"],
                    label=cfg["label"],
                )
            )

    # Statistical anomaly detector (added later after training)
    return validators


# ── Main simulation ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print(BANNER)
    t_start = time.perf_counter()

    # Override config from CLI flags
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
    # Generate enough normal routes so that after mixing with attacks
    # we end up with approximately args.routes total announcements
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
    print(f"    Decision threshold  : {anomaly.threshold}")

    # ── Step 5: Build Validators ─────────────────────────────────────────────
    _print_section("Step 5 – Building Validators")
    validators = build_validators(topo)
    # Deduplicate (RPKI validator is also inside HOP_COMBINATIONS via 'origin_only')
    seen_names: set = set()
    unique_validators = []
    for v in validators:
        if v.name not in seen_names:
            unique_validators.append(v)
            seen_names.add(v.name)
    unique_validators.append(anomaly)

    # Composite: Selective hop (origin+first+last) AND anomaly detector
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

    # Find the origin+first+last validator
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
    pd.set_option("display.width",        120)
    print(summary[["detection_rate", "false_positive_rate", "avg_time_ms", "security_cost_ratio"]].to_string())

    _print_section("Results – Detection Rate by Attack Type")
    print(breakdown.to_string())

    # Key findings
    _print_section("Key Findings")
    ofl_name = "Origin + 1st + Last"
    anom_name = "Anomaly Detector"
    bgpsec_name = "BGPsec (Full Path)"
    composite_name = "Selective + Anomaly"

    for vname, label in [
        (ofl_name,       "Selective hop (origin+first+last)"),
        (anom_name,      "Statistical anomaly detection"),
        (bgpsec_name,    "BGPsec (full path)"),
        (composite_name, "Combined (selective + anomaly)"),
    ]:
        if vname in summary.index:
            dr = summary.loc[vname, "detection_rate"] * 100
            fpr = summary.loc[vname, "false_positive_rate"] * 100
            t   = summary.loc[vname, "avg_time_ms"] * 1000
            print(f"  {label}")
            print(f"      Detection Rate  : {dr:.1f}%")
            print(f"      False Pos. Rate : {fpr:.2f}%")
            print(f"      Avg. Time       : {t:.3f} µs/announcement")

    # ── Step 8: Figures ───────────────────────────────────────────────────────
    if not args.no_plots:
        _print_section("Step 8 – Generating Figures")
        from src.visualization import generate_all_figures
        generate_all_figures(summary, breakdown)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'─'*64}")
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"{'─'*64}\n")

    # Save summary CSV
    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary.csv"))
    breakdown.to_csv(os.path.join(config.RESULTS_DIR, "attack_breakdown.csv"))
    print(f"  CSV results saved to ./{config.RESULTS_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
