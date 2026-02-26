"""
Evaluator
=========
Runs all validators against a mixed traffic stream and computes the key
security-performance metrics:

  • Detection Rate (True Positive Rate)  – % of attacks caught
  • False Positive Rate                  – % of legit routes wrongly flagged
  • Average Processing Time (ms)         – cost per announcement
  • Security-Cost Ratio                  – detection_rate / avg_processing_time

Also supports per-attack-type breakdown and per-validator comparison tables
(returned as pandas DataFrames for easy analysis and export).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.bgp_simulator import BGPRoute
from src.validators.base import BaseValidator


class ValidationResult:
    """Stores the outcome of running one validator against one route."""
    __slots__ = ("route", "accepted", "processing_time_ms")

    def __init__(self, route: BGPRoute, accepted: bool, processing_time_ms: float) -> None:
        self.route               = route
        self.accepted            = accepted
        self.processing_time_ms  = processing_time_ms


class Evaluator:
    """
    Benchmark multiple validators against the same traffic stream.

    Parameters
    ----------
    validators : list[BaseValidator]
        All validators to benchmark.
    routes : list[BGPRoute]
        Mixed traffic (legitimate + attack routes).
    """

    def __init__(
        self,
        validators: List[BaseValidator],
        routes: List[BGPRoute],
    ) -> None:
        self.validators = validators
        self.routes     = routes
        self._results:  Dict[str, List[ValidationResult]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Run every validator against the full traffic stream."""
        for v in self.validators:
            print(f"  Evaluating: {v.name}")
            self._results[v.name] = self._run_one(v)

    def summary_table(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per validator and columns:
          validator, detection_rate, false_positive_rate,
          avg_time_ms, total_time_ms, security_cost_ratio
        """
        rows = []
        for v in self.validators:
            name = v.name
            if name not in self._results:
                continue
            metrics = self._compute_metrics(self._results[name])
            metrics["validator"] = name
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("validator")
        col_order = [
            "detection_rate", "false_positive_rate",
            "avg_time_ms", "total_time_ms", "security_cost_ratio",
            "n_attacks", "n_legit", "n_detected", "n_fp",
        ]
        return df[[c for c in col_order if c in df.columns]]

    def attack_type_breakdown(self) -> pd.DataFrame:
        """
        Return a DataFrame: rows = validators, columns = attack types,
        values = detection rate for that attack type.
        """
        attack_types = sorted({
            r.route.attack_type
            for results in self._results.values()
            for r in results
            if r.route.is_attack and r.route.attack_type
        })

        rows = []
        for v in self.validators:
            name = v.name
            if name not in self._results:
                continue
            row = {"validator": name}
            for at in attack_types:
                at_results = [
                    r for r in self._results[name]
                    if r.route.is_attack and r.route.attack_type == at
                ]
                if at_results:
                    detected = sum(1 for r in at_results if not r.accepted)
                    row[at] = round(detected / len(at_results), 4)
                else:
                    row[at] = float("nan")
            rows.append(row)

        return pd.DataFrame(rows).set_index("validator")

    def selective_hop_comparison(self, hop_validators: List[BaseValidator]) -> pd.DataFrame:
        """
        Detailed comparison of selective-hop validators: shows how detection
        rate varies with number of verified hops (cost proxy).
        """
        rows = []
        for v in hop_validators:
            name = v.name
            if name not in self._results:
                continue
            metrics = self._compute_metrics(self._results[name])
            # Approximate #hops verified from average processing time
            n_hops = round(metrics["avg_time_ms"] / 0.5, 2) if metrics["avg_time_ms"] > 0 else 0
            rows.append({
                "validator":      name,
                "avg_hops_verified": n_hops,
                "detection_rate": metrics["detection_rate"],
                "false_positive_rate": metrics["false_positive_rate"],
                "avg_time_ms":    metrics["avg_time_ms"],
            })
        return pd.DataFrame(rows).set_index("validator")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_one(self, validator: BaseValidator) -> List[ValidationResult]:
        results = []
        for route in tqdm(self.routes, desc=f"    {validator.name}", leave=False, unit="route"):
            accepted, t_ms = validator.validate(route)
            results.append(ValidationResult(route, accepted, t_ms))
        return results

    def _compute_metrics(self, results: List[ValidationResult]) -> Dict:
        attacks = [r for r in results if r.route.is_attack]
        legit   = [r for r in results if not r.route.is_attack]

        n_attacks = len(attacks)
        n_legit   = len(legit)

        # Detected = attack route that was rejected (accepted=False)
        n_detected = sum(1 for r in attacks if not r.accepted)
        # False positives = legit route that was rejected
        n_fp       = sum(1 for r in legit if not r.accepted)

        detection_rate      = n_detected / n_attacks if n_attacks else 0.0
        false_positive_rate = n_fp       / n_legit   if n_legit   else 0.0

        times      = [r.processing_time_ms for r in results]
        avg_time   = float(np.mean(times))
        total_time = float(np.sum(times))

        # Security-cost ratio: detection rate per ms of average processing
        sc_ratio = detection_rate / avg_time if avg_time > 0 else 0.0

        return {
            "detection_rate":      round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "avg_time_ms":         round(avg_time, 6),
            "total_time_ms":       round(total_time, 2),
            "security_cost_ratio": round(sc_ratio, 4),
            "n_attacks":           n_attacks,
            "n_legit":             n_legit,
            "n_detected":          n_detected,
            "n_fp":                n_fp,
        }
