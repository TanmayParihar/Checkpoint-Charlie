"""
Enhanced Checkpoint Charlie Pipeline
====================================
Adds ASPA, OTC, GNN anomaly detection, path plausibility fusion, and
route-leak attacks without modifying the original pipeline files.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from src.attack_injector import AttackInjector
from src.attacks.route_leak import RouteLeakInjector
from src.bgp_simulator import BGPRoute, BGPSimulator
from src.evaluator import Evaluator
from src.topology import ASTopology
from src.validators.anomaly import AnomalyDetector
from src.validators.aspa import ASPAValidator
from src.validators.bgpsec import BGPsecValidator
from src.validators.delta_trust import DeltaTrustValidator
from src.validators.gnn_anomaly import GNNAnomalyDetector
from src.validators.otc import OTCValidator
from src.validators.path_plausibility import PathPlausibilityValidator
from src.validators.rpki import RPKIValidator
from src.validators.selective_hop import SelectiveHopValidator


BANNER = """
======================================================================
  Checkpoint Charlie Enhanced Benchmark
  ASPA + OTC + GNN + Path Plausibility + Route Leak Modeling
======================================================================
"""


class SelectiveAnomalyComposite:
    """Flags route if either selective-hop or anomaly detector flags it."""

    name = "Selective + Anomaly"

    def __init__(self, hop_v, anomaly_v):
        self._hop = hop_v
        self._anomaly = anomaly_v

    def validate(self, route: BGPRoute):
        h_ok, h_t = self._hop.validate(route)
        a_ok, a_t = self._anomaly.validate(route)
        return (h_ok and a_ok), (h_t + a_t)


class PartialRPKIValidator(RPKIValidator):
    """RPKI validator with synthetic partial-coverage masking."""

    def __init__(self, topology: ASTopology, coverage: float, seed: int = 42):
        super().__init__(topology)
        self.coverage = max(0.0, min(1.0, coverage))
        rng = random.Random(seed)
        subset = {}
        for prefix, origin in self.roa_db.items():
            if rng.random() <= self.coverage:
                subset[prefix] = origin
        self.roa_db = subset
        self.name = f"RPKI ({int(self.coverage * 100)}%)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced BGP security simulation.")
    p.add_argument("--ases", type=int, default=config.NUM_ASES)
    p.add_argument("--routes", type=int, default=config.NUM_ROUTES)
    p.add_argument("--attack-rate", type=float, default=config.ATTACK_RATE)
    p.add_argument("--route-leak-share", type=float, default=0.35, help="Fraction of attacks that are route leaks")
    p.add_argument("--rpki-cov", type=float, default=config.RPKI_COVERAGE)
    p.add_argument("--aspa-cov", type=float, default=0.50)
    p.add_argument("--otc-cov", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--sweep-sample", type=int, default=5000)
    p.add_argument("--results-dir", type=str, default="results_enhanced")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _print_section(title: str) -> None:
    print(f"\n{'-' * 72}")
    print(f"  {title}")
    print(f"{'-' * 72}")


def _inject_classic_attacks(
    topology: ASTopology,
    normal_routes: Sequence[BGPRoute],
    n_attacks: int,
    attack_dist: Dict[str, float],
    seed: int,
) -> List[BGPRoute]:
    if not normal_routes or n_attacks <= 0:
        return []

    injector = AttackInjector(
        topology=topology,
        attack_rate=0.0,
        attack_type_dist=attack_dist,
        seed=seed,
    )
    attack_types = list(attack_dist.keys())
    probs = [attack_dist[k] for k in attack_types]

    out: List[BGPRoute] = []
    for _ in range(n_attacks):
        base = copy.deepcopy(injector.rng.choice(normal_routes))
        atype = injector.rng.choices(attack_types, weights=probs, k=1)[0]
        attacked = injector._inject(base, atype)
        if attacked is not None:
            out.append(attacked)
    return out


def _traffic_stats(routes: Sequence[BGPRoute]) -> Dict:
    total = len(routes)
    attacks = [r for r in routes if r.is_attack]
    by_type: Dict[str, int] = {}
    for route in attacks:
        by_type[route.attack_type or "unknown"] = by_type.get(route.attack_type or "unknown", 0) + 1
    return {
        "total": total,
        "legitimate": total - len(attacks),
        "attacks": len(attacks),
        "attack_rate": (len(attacks) / max(1, total)),
        "by_type": by_type,
    }


def _sample_routes(routes: Sequence[BGPRoute], sample_size: int, seed: int) -> List[BGPRoute]:
    if sample_size <= 0 or sample_size >= len(routes):
        return list(routes)
    rng = random.Random(seed)
    return rng.sample(list(routes), sample_size)


def _metrics_for_validator(validator, routes: Sequence[BGPRoute]) -> Dict[str, float]:
    attacks = 0
    legit = 0
    detected = 0
    fp = 0
    times: List[float] = []

    for route in routes:
        accepted, t_ms = validator.validate(route)
        times.append(t_ms)
        if route.is_attack:
            attacks += 1
            if not accepted:
                detected += 1
        else:
            legit += 1
            if not accepted:
                fp += 1

    detection = detected / max(1, attacks)
    fpr = fp / max(1, legit)
    return {
        "detection_rate": detection,
        "false_positive_rate": fpr,
        "avg_time_ms": float(np.mean(times)) if times else 0.0,
    }


def _binary_detection_rate(validator, routes: Sequence[BGPRoute]) -> float:
    if not routes:
        return 0.0
    detected = 0
    for route in routes:
        accepted, _ = validator.validate(route)
        if not accepted:
            detected += 1
    return detected / len(routes)


def _aspa_coverage_sweep(
    topology: ASTopology,
    leak_routes: Sequence[BGPRoute],
    coverage_points: Sequence[float],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for i, cov in enumerate(coverage_points):
        v = ASPAValidator(topology, coverage=cov, seed=seed + i)
        det = _binary_detection_rate(v, leak_routes)
        rows.append({"coverage": cov, "detection_rate": det, "n_routes": len(leak_routes)})
    return pd.DataFrame(rows)


def _partial_deployment_curve(
    topology: ASTopology,
    routes: Sequence[BGPRoute],
    coverage_points: Sequence[float],
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for i, cov in enumerate(coverage_points):
        rpki = PartialRPKIValidator(topology, coverage=cov, seed=seed + i)
        aspa = ASPAValidator(topology, coverage=cov, seed=seed + 100 + i)
        otc = OTCValidator(topology, adoption_rate=cov, seed=seed + 200 + i)

        for validator in (rpki, aspa, otc):
            metrics = _metrics_for_validator(validator, routes)
            rows.append(
                {
                    "coverage": cov,
                    "validator": validator.name.split(" (")[0],
                    "detection_rate": metrics["detection_rate"],
                }
            )
    return pd.DataFrame(rows)


def _fusion_ablation(
    topology: ASTopology,
    routes: Sequence[BGPRoute],
    rpki,
    selective,
    anomaly,
    gnn,
    *,
    aspa_cov: float,
    otc_cov: float,
    seed: int,
) -> pd.DataFrame:
    scenarios: List[Tuple[str, set]] = []
    all_signals = set(PathPlausibilityValidator.SIGNAL_NAMES)
    scenarios.append(("All signals", all_signals))
    for signal in sorted(all_signals):
        scenarios.append((f"No {signal}", all_signals - {signal}))

    rows = []
    for i, (label, enabled) in enumerate(scenarios):
        aspa = ASPAValidator(topology, coverage=aspa_cov, seed=seed + 10 + i)
        otc = OTCValidator(topology, adoption_rate=otc_cov, seed=seed + 20 + i)
        delta = DeltaTrustValidator(topology, label=f"Delta Fusion Ablation {i}")
        fusion = PathPlausibilityValidator(
            rpki,
            aspa,
            otc,
            selective,
            anomaly,
            gnn,
            delta,
            enabled_signals=enabled,
        )
        metrics = _metrics_for_validator(fusion, routes)
        rows.append(
            {
                "scenario": label,
                "detection_rate": metrics["detection_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
                "avg_time_ms": metrics["avg_time_ms"],
            }
        )
    return pd.DataFrame(rows)


def _roc_dataframe(y_true: Sequence[int], score_map: Dict[str, Sequence[float]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for name, scores in score_map.items():
        fpr, tpr, auc_value = _compute_roc(y_true, scores)
        for x, y in zip(fpr, tpr):
            rows.append({"validator": name, "fpr": x, "tpr": y, "auc": auc_value})
    return pd.DataFrame(rows)


def _compute_roc(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[List[float], List[float], float]:
    try:
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_true, scores)
        return list(fpr), list(tpr), float(auc(fpr, tpr))
    except Exception:
        # Fallback implementation when sklearn is unavailable.
        thresholds = sorted(set(float(s) for s in scores), reverse=True)
        thresholds = [1.1] + thresholds + [-0.1]
        fpr: List[float] = []
        tpr: List[float] = []

        positives = sum(1 for y in y_true if y == 1)
        negatives = sum(1 for y in y_true if y == 0)
        positives = max(1, positives)
        negatives = max(1, negatives)

        for thr in thresholds:
            tp = fp = 0
            for y, s in zip(y_true, scores):
                pred = 1 if s >= thr else 0
                if pred == 1 and y == 1:
                    tp += 1
                elif pred == 1 and y == 0:
                    fp += 1
            tpr.append(tp / positives)
            fpr.append(fp / negatives)

        # Sort by FPR for area computation.
        pairs = sorted(zip(fpr, tpr), key=lambda p: p[0])
        fpr = [p[0] for p in pairs]
        tpr = [p[1] for p in pairs]
        auc_value = float(np.trapz(y=tpr, x=fpr))
        return fpr, tpr, auc_value


def run(args: argparse.Namespace) -> None:
    print(BANNER)
    t_start = time.perf_counter()
    config.RESULTS_DIR = args.results_dir

    # Step 1: topology
    _print_section("Step 1 - Building AS Topology")
    topo = ASTopology(
        num_ases=args.ases,
        tier1_count=config.TIER1_COUNT,
        tier2_count=config.TIER2_COUNT,
        rpki_coverage=args.rpki_cov,
        seed=args.seed,
    )
    for k, v in topo.stats().items():
        print(f"    {k:<25} {v}")

    # Step 2: normal routes
    _print_section("Step 2 - Generating Normal Routes")
    n_normal = int(args.routes * (1 - args.attack_rate))
    n_target_attacks = max(0, args.routes - n_normal)

    sim = BGPSimulator(topo, num_routes=n_normal, seed=args.seed)
    normal_routes = sim.generate_normal_routes(n_normal)
    print(f"    Generated {len(normal_routes):,} normal routes")

    # Step 3: attacks (classic + route leaks)
    _print_section("Step 3 - Injecting Classic Attacks + Route Leaks")
    leak_share = max(0.0, min(1.0, args.route_leak_share))
    n_leaks = int(n_target_attacks * leak_share)
    n_classic = max(0, n_target_attacks - n_leaks)

    classic_attacks = _inject_classic_attacks(
        topo,
        normal_routes,
        n_attacks=n_classic,
        attack_dist=config.ATTACK_TYPE_DIST,
        seed=args.seed,
    )

    leak_injector = RouteLeakInjector(topo, seed=args.seed + 77)
    leak_attacks = leak_injector.inject_n_leaks(normal_routes, n_leaks)

    all_routes = list(normal_routes) + classic_attacks + leak_attacks
    # Backfill if some attacks could not be generated.
    missing = max(0, args.routes - len(all_routes))
    if missing > 0:
        all_routes.extend(
            _inject_classic_attacks(
                topo,
                normal_routes,
                n_attacks=missing,
                attack_dist=config.ATTACK_TYPE_DIST,
                seed=args.seed + 999,
            )
        )

    random.Random(args.seed).shuffle(all_routes)
    stats = _traffic_stats(all_routes)
    print(f"    Total announcements : {stats['total']:,}")
    print(f"    Legitimate          : {stats['legitimate']:,}")
    print(f"    Attacks             : {stats['attacks']:,} ({stats['attack_rate']*100:.1f}%)")
    for attack_type, cnt in sorted(stats["by_type"].items()):
        print(f"      - {attack_type:<30} {cnt:,}")

    # Step 4: train anomaly detectors
    _print_section("Step 4 - Training Anomaly Models")
    anomaly = AnomalyDetector(topo, threshold=config.ANOMALY_SCORE_THRESHOLD)
    anomaly.train(normal_routes)
    gnn = GNNAnomalyDetector(topo, threshold=0.58, engine="auto")
    gnn.train(normal_routes)
    print(f"    Statistical detector trained on {len(normal_routes):,} routes")
    print(f"    GNN detector engine: {gnn.engine}")

    # Step 5: build validators
    _print_section("Step 5 - Building Validators")
    rpki = RPKIValidator(topo)
    delta = DeltaTrustValidator(topo, label="Delta Trust")
    aspa = ASPAValidator(topo, coverage=args.aspa_cov, seed=args.seed + 1)
    aspa.name = f"ASPA ({int(args.aspa_cov * 100)}%)"
    otc = OTCValidator(topo, adoption_rate=args.otc_cov, seed=args.seed + 2)
    otc.name = f"OTC ({int(args.otc_cov * 100)}%)"

    validators: List = [rpki]
    selective_ofl = None
    for cfg in config.HOP_COMBINATIONS.values():
        if cfg["positions"] is None:
            validators.append(BGPsecValidator(topo))
        else:
            v = SelectiveHopValidator(topo, positions=cfg["positions"], label=cfg["label"])
            validators.append(v)
            if "1st + Last" in v.name:
                selective_ofl = v

    validators.append(anomaly)
    validators.append(delta)
    validators.append(aspa)
    validators.append(otc)
    validators.append(gnn)

    if selective_ofl is not None:
        validators.append(SelectiveAnomalyComposite(selective_ofl, anomaly))
    else:
        selective_ofl = SelectiveHopValidator(topo, positions=[0, 1, -1], label="Origin + 1st + Last")
        validators.append(selective_ofl)

    fusion_delta = DeltaTrustValidator(topo, label="Delta (Fusion Internal)")
    path_fusion = PathPlausibilityValidator(
        rpki_validator=rpki,
        aspa_validator=aspa,
        otc_validator=otc,
        selective_validator=selective_ofl,
        anomaly_validator=anomaly,
        gnn_validator=gnn,
        delta_validator=fusion_delta,
    )
    validators.append(path_fusion)

    seen = set()
    unique_validators = []
    for v in validators:
        if v.name not in seen:
            unique_validators.append(v)
            seen.add(v.name)
            print(f"    - {v.name}")

    # Step 6: evaluate
    _print_section("Step 6 - Evaluating All Validators")
    evaluator = Evaluator(unique_validators, all_routes)
    evaluator.run_all()
    summary = evaluator.summary_table()
    breakdown = evaluator.attack_type_breakdown()

    _print_section("Results - Summary")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary[["detection_rate", "false_positive_rate", "avg_time_ms", "security_cost_ratio"]].to_string())

    # Step 7: advanced analysis
    _print_section("Step 7 - Enhanced Analysis")
    analysis_routes = _sample_routes(all_routes, args.sweep_sample, args.seed + 7)
    leak_routes = [r for r in analysis_routes if r.is_attack and str(r.attack_type).startswith("route_leak")]
    if not leak_routes:
        leak_routes = [r for r in all_routes if r.is_attack and str(r.attack_type).startswith("route_leak")]

    coverage_points = [0.05, 0.25, 0.50, 0.75, 1.00]
    aspa_sweep_df = _aspa_coverage_sweep(topo, leak_routes, coverage_points, args.seed + 50)
    partial_curve_df = _partial_deployment_curve(topo, analysis_routes, coverage_points, args.seed + 60)
    ablation_df = _fusion_ablation(
        topo,
        analysis_routes,
        rpki=rpki,
        selective=selective_ofl,
        anomaly=anomaly,
        gnn=gnn,
        aspa_cov=args.aspa_cov,
        otc_cov=args.otc_cov,
        seed=args.seed + 70,
    )

    # ROC data (probabilistic validators).
    y_true = [1 if r.is_attack else 0 for r in analysis_routes]
    gnn_scores = [gnn.score_route(r) for r in analysis_routes]  # anomaly probability

    roc_fusion = PathPlausibilityValidator(
        rpki_validator=rpki,
        aspa_validator=ASPAValidator(topo, coverage=args.aspa_cov, seed=args.seed + 81),
        otc_validator=OTCValidator(topo, adoption_rate=args.otc_cov, seed=args.seed + 82),
        selective_validator=selective_ofl,
        anomaly_validator=anomaly,
        gnn_validator=gnn,
        delta_validator=DeltaTrustValidator(topo, label="Delta ROC Internal"),
    )
    fusion_scores = []
    for route in analysis_routes:
        score, _, _, _ = roc_fusion.score_route(route)
        fusion_scores.append(1.0 - score)  # convert plausibility -> anomaly probability

    roc_df = _roc_dataframe(
        y_true,
        {
            "GNN Anomaly Detector": gnn_scores,
            "Path Plausibility": fusion_scores,
        },
    )

    # Leak heatmap subset.
    leak_cols = [c for c in breakdown.columns if str(c).startswith("route_leak")]
    leak_heatmap_df = breakdown[leak_cols] if leak_cols else pd.DataFrame()

    # Step 8: figures and CSVs
    if not args.no_plots:
        _print_section("Step 8 - Generating Figures")
        from src.visualization import generate_all_figures
        from src.enhanced_visualization import generate_enhanced_figures

        generate_all_figures(summary, breakdown)
        generate_enhanced_figures(
            summary,
            breakdown,
            results_dir=args.results_dir,
            aspa_sweep=aspa_sweep_df,
            partial_curve=partial_curve_df,
            ablation=ablation_df,
            leak_heatmap=leak_heatmap_df,
            roc_curves=roc_df,
        )

    os.makedirs(args.results_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"))
    breakdown.to_csv(os.path.join(args.results_dir, "attack_breakdown.csv"))
    aspa_sweep_df.to_csv(os.path.join(args.results_dir, "aspa_coverage_sweep.csv"), index=False)
    partial_curve_df.to_csv(os.path.join(args.results_dir, "partial_deployment_curve.csv"), index=False)
    ablation_df.to_csv(os.path.join(args.results_dir, "fusion_ablation.csv"), index=False)
    leak_heatmap_df.to_csv(os.path.join(args.results_dir, "route_leak_heatmap.csv"))
    roc_df.to_csv(os.path.join(args.results_dir, "roc_curves.csv"), index=False)

    elapsed = time.perf_counter() - t_start
    _print_section("Completed")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Results: ./{args.results_dir}/")


if __name__ == "__main__":
    run(parse_args())
