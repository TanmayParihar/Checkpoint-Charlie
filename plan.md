# Checkpoint Charlie — Next-Generation BGP Path Security Enhancement Plan

> **Date:** March 25, 2026
> **Scope:** New modules only — all existing files remain untouched
> **Goal:** Implement the latest, most efficient BGP path security techniques informed by cutting-edge research and emerging IETF standards (2025-2026)

---

## 1. Executive Summary

The current Checkpoint Charlie project simulates BGP path security using RPKI ROV, Selective Hop Verification, BGPsec (full path), a Statistical Anomaly Detector, and a Delta Trust Validator. These are strong foundations but do **not** yet include:

1. **ASPA (Autonomous System Provider Authorization)** — the IETF's emerging post-ROV standard that is now in early production deployment (ARIN & RIPE NCC, 2026).
2. **OTC (Only-to-Customer) Route Leak Detection** — RFC 9234's lightweight attribute-based leak prevention.
3. **Graph Neural Network (GNN) Anomaly Detection** — topology-aware ML that dramatically outperforms traditional statistical detectors.
4. **Path Plausibility Scoring** — a unified probabilistic framework that fuses all signals (RPKI, ASPA, topology, ML) into a single trust score per route.
5. **Route Leak Attacks** — a critical missing attack class that ASPA and OTC specifically target.

This plan adds all five as **new files**, benchmarks them against the existing validators, and produces an enhanced comparative evaluation.

---

## 2. What We're Building (New Files Only)

### 2.1 ASPA Validator
**File:** `src/validators/aspa.py`

| Aspect | Detail |
|--------|--------|
| **What** | Implements ASPA-based AS_PATH verification per [draft-ietf-sidrops-aspa-verification-24](https://datatracker.ietf.org/doc/draft-ietf-sidrops-aspa-verification/) |
| **How** | Each AS publishes a set of authorized provider ASNs. For each route, the validator walks the AS_PATH and checks each hop-pair relationship against the ASPA database. Returns **Valid**, **Invalid**, or **Unknown** |
| **Verification Algorithm** | **Upstream verification:** Walk AS_PATH from origin → receiver. Each AS must list the next AS as an authorized provider (or be peers). If any hop has the next AS listed as *not* a provider and it's not a peer, mark Invalid. **Downstream verification:** Walk AS_PATH from receiver → origin with analogous checks |
| **Attack Coverage** | Route leaks (primary), forged-origin hijacks, some path fabrication |
| **Overhead** | Hash-table lookups only — **no cryptographic signatures**. Expected ~0.02-0.05 ms per route |
| **Partial Deployment** | Configurable ASPA coverage parameter (e.g., 5%, 25%, 50%) to simulate real-world incremental rollout |
| **Key Advantage** | Catches route leaks that **neither RPKI ROV nor BGPsec can detect**. Lightweight enough for immediate deployment |

**Research Basis:**
- IETF draft-ietf-sidrops-aspa-verification-24 (expires April 2026)
- NDSS 2025: "Securing BGP ASAP: ASPA and other Post-ROV Defenses" (Furuness et al.)
- NIST BRIO test framework (August 2025)
- NIST SP 800-189 Rev. 1 (January 2025)

---

### 2.2 OTC Route Leak Detector
**File:** `src/validators/otc.py`

| Aspect | Detail |
|--------|--------|
| **What** | Implements RFC 9234 "Only-to-Customer" attribute-based route leak detection |
| **How** | Uses BGP Role negotiation (Provider, Customer, Peer, RS, RS-Client) and the OTC transitive attribute. If a route with OTC set arrives from a customer, it is a leak |
| **Overhead** | Single attribute check — ~0.01 ms per route |
| **Partial Deployment** | Configurable OTC adoption rate to simulate incremental rollout |
| **Key Advantage** | Simplest possible route leak detection; works even without RPKI |

**Research Basis:**
- RFC 9234 (June 2022, now in deployment)
- Höger 2025: "Mitigating BGP Route Leaks With Attributes and Communities"

---

### 2.3 GNN-Based Anomaly Detector
**File:** `src/validators/gnn_anomaly.py`

| Aspect | Detail |
|--------|--------|
| **What** | Graph Neural Network that operates on the AS topology graph to detect anomalous BGP announcements |
| **How** | Constructs a graph where nodes = ASes, edges = peering/transit relationships. Each route announcement is embedded as a subgraph (the AS_PATH overlaid on the topology). A GNN classifies each route as legitimate or anomalous |
| **Architecture** | 2-layer Graph Attention Network (GAT) with node features: AS degree, tier, historical announcement count, prefix diversity. Edge features: relationship type, age |
| **Training** | Trained on the normal route history (same as existing anomaly detector) using a semi-supervised approach |
| **Expected Performance** | ~90-96% detection accuracy (per GNN BGP research), significantly better than the existing 76.16% statistical detector |
| **Overhead** | Model inference ~0.5-2.0 ms per route (GPU) or ~5-10 ms (CPU). We'll benchmark both |
| **Key Advantage** | Topology-aware — understands graph structure, not just statistical features. Catches subtle path fabrication that statistical methods miss |

**Research Basis:**
- "Unveiling the Potential of Graph Neural Networks for BGP Anomaly Detection" (ACM GNNet '22 / CoNEXT)
- "BGNN: Detection of BGP Anomalies Using Graph Neural Networks" (HAL, 2022)
- "A Multi-View Framework for BGP Anomaly Detection via Graph Attention Network" (ScienceDirect, 2022)
- Springer 2026: "Graph Neural Networks for Anomaly Detection: A Systematic Review of Dynamic Temporal Approaches"

---

### 2.4 Path Plausibility Scoring Engine
**File:** `src/validators/path_plausibility.py`

| Aspect | Detail |
|--------|--------|
| **What** | A unified probabilistic framework that fuses ALL available signals into a single plausibility score per route |
| **How** | Weighted Bayesian fusion of: RPKI ROV result (Valid/Invalid/Unknown), ASPA verification state, OTC leak signal, Selective Hop adjacency checks, Statistical anomaly score, GNN anomaly score, Delta Trust cache-hit ratio |
| **Output** | Continuous score 0.0 (certainly malicious) → 1.0 (certainly legitimate), plus categorical verdict (Accept / Suspect / Reject) with configurable thresholds |
| **Key Innovation** | No existing tool combines all these signals probabilistically. Each signal has different false-positive/false-negative profiles; fusion yields superior accuracy |
| **Overhead** | Sum of component overheads + ~0.01 ms fusion computation |
| **Configurable** | Weights per signal can be tuned; supports ablation studies to measure each signal's marginal contribution |

**Research Basis:**
- Novel contribution synthesizing ASPA + ROV + ML approaches
- Inspired by NIST SP 800-189 Rev. 1 layered defense recommendations

---

### 2.5 Route Leak Attack Injector
**File:** `src/attacks/route_leak.py`

| Aspect | Detail |
|--------|--------|
| **What** | Adds route leak attacks to the simulation — the #1 real-world BGP incident type that the current project doesn't model |
| **Attack Types** | **Type 1 — Full Prefix Leak:** Customer leaks provider's full table to another provider. **Type 2 — Lateral Leak:** Peer leaks peer-learned routes to a provider. **Type 3 — Intentional Leak:** Attacker strategically leaks to attract traffic |
| **Why Missing Matters** | ASPA and OTC are specifically designed to catch route leaks. Without this attack class, we can't properly evaluate them |
| **Integration** | Plugs into the existing attack pipeline alongside origin hijack, path shortening, and path fabrication |

---

### 2.6 Enhanced Evaluation & Visualization
**Files:**
- `enhanced_main.py` — New top-level entry point that runs ALL validators (existing + new)
- `src/enhanced_visualization.py` — New visualization module with additional plots

| New Plot | Purpose |
|----------|---------|
| ASPA Coverage Sweep | Detection rate vs. ASPA deployment percentage (5%-100%) |
| Partial Deployment Curve | How each validator performs as deployment grows |
| Signal Fusion Ablation | Marginal contribution of each signal in the Path Plausibility engine |
| Route Leak Detection Heatmap | Per-validator detection of each route leak subtype |
| Radar/Spider Chart | Multi-dimensional comparison (detection, FPR, overhead, leak coverage, partial-deploy robustness) |
| ROC Curves | Full ROC curves for probabilistic validators (GNN, Plausibility) |

---

## 3. Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │         enhanced_main.py            │
                          │  (orchestrates full pipeline)       │
                          └──────────────┬──────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
  ┌───────────────┐           ┌──────────────────┐          ┌──────────────────┐
  │  EXISTING     │           │  NEW VALIDATORS  │          │  NEW ATTACKS     │
  │  (untouched)  │           │  (new files)     │          │  (new file)      │
  ├───────────────┤           ├──────────────────┤          ├──────────────────┤
  │ rpki.py       │           │ aspa.py          │          │ route_leak.py    │
  │ bgpsec.py     │           │ otc.py           │          └──────────────────┘
  │ selective_hop │           │ gnn_anomaly.py   │
  │ anomaly.py    │           │ path_plausibility│
  │ delta_trust   │           └──────────────────┘
  └───────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌──────────────────┐
              │   Evaluator      │
              │   (existing,     │
              │    untouched)    │
              └────────┬─────────┘
                       ▼
           ┌───────────────────────┐
           │ enhanced_visualization│
           │ (new file)            │
           └───────────────────────┘
```

---

## 4. Expected Outcomes & Metrics

### 4.1 Detection Rate Projections

| Validator | Origin Hijack | Path Shortening | Path Fabrication | Route Leak | Overall |
|-----------|:---:|:---:|:---:|:---:|:---:|
| RPKI (existing) | ~45% | 0% | 0% | 0% | ~19% |
| Anomaly (existing) | ~80% | ~64% | ~75% | ~40% | ~76% |
| Selective Hop O+1+L (existing) | ~99% | ~90% | ~88% | 0% | ~95% |
| BGPsec (existing) | ~99% | ~99% | 100% | 0% | ~98% |
| **ASPA (new)** | **~50-70%** | **~30%** | **~20%** | **~85-95%** | **~60-75%** |
| **OTC (new)** | **0%** | **0%** | **0%** | **~70-90%** | **~20-30%** |
| **GNN Anomaly (new)** | **~92%** | **~85%** | **~90%** | **~75%** | **~90-96%** |
| **Path Plausibility (new)** | **~99%** | **~97%** | **~98%** | **~95%** | **~97-99%** |

### 4.2 Efficiency Projections

| Validator | Avg Time (ms) | Security-Cost Ratio |
|-----------|:---:|:---:|
| RPKI (existing) | 0.231 | 0.82 |
| Anomaly (existing) | 0.050 | 15.23 |
| BGPsec (existing) | 1.732 | 0.57 |
| **ASPA (new)** | **~0.03** | **~20-25** |
| **OTC (new)** | **~0.01** | **~20-30** |
| **GNN Anomaly (new)** | **~2.0** | **~0.45** |
| **Path Plausibility (new)** | **~3.0** | **~0.33** |

### 4.3 Key Hypotheses to Validate

1. **ASPA has the best security-cost ratio for route leak detection** — near-zero overhead with high leak coverage
2. **GNN outperforms the statistical anomaly detector by ≥15 percentage points** while remaining cryptography-free
3. **Path Plausibility fusion achieves ≥97% overall detection** with acceptable overhead — better than any single validator
4. **Partial deployment of ASPA at just 25% coverage** still provides meaningful route leak protection (>50% detection)
5. **The combination of ASPA + OTC + existing anomaly detector** provides >90% detection at <0.1 ms — the "sweet spot" for practical deployment

---

## 5. Implementation Order & Dependencies

```
Phase 1 — Foundation (no dependencies)
  ├── 1a. Route Leak Attack Injector     (src/attacks/route_leak.py)
  ├── 1b. OTC Validator                  (src/validators/otc.py)
  └── 1c. ASPA Validator                 (src/validators/aspa.py)

Phase 2 — ML Layer (depends on topology infrastructure)
  └── 2a. GNN Anomaly Detector           (src/validators/gnn_anomaly.py)

Phase 3 — Fusion (depends on all above)
  └── 3a. Path Plausibility Engine       (src/validators/path_plausibility.py)

Phase 4 — Integration & Visualization
  ├── 4a. Enhanced Main                  (enhanced_main.py)
  └── 4b. Enhanced Visualization         (src/enhanced_visualization.py)
```

---

## 6. New Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `torch` | PyTorch for GNN inference | ≥2.0 |
| `torch_geometric` | PyTorch Geometric for graph neural networks | ≥2.4 |
| `scikit-learn` | ROC curves, metrics utilities | ≥1.3 |

> These will be added to a **new** `requirements_enhanced.txt` file (existing `requirements.txt` remains untouched).

---

## 7. Research Sources & References

### IETF Standards & Drafts
- [ASPA Verification Draft (draft-ietf-sidrops-aspa-verification-24)](https://datatracker.ietf.org/doc/draft-ietf-sidrops-aspa-verification/)
- [ASPA Profile Draft (draft-ietf-sidrops-aspa-profile-22)](https://datatracker.ietf.org/doc/draft-ietf-sidrops-aspa-profile/)
- [RFC 9234 — Route Leak Prevention Using Roles (OTC)](https://www.rfc-editor.org/rfc/rfc9234.html)
- [BGP Open Policy Draft](https://datatracker.ietf.org/doc/draft-ietf-idr-bgp-open-policy/)

### NIST Publications
- [NIST SP 800-189 Rev. 1 — BGP Security and Resilience (Jan 2025)](https://csrc.nist.gov/pubs/sp/800/189/r1/ipd)
- [NIST BRIO Test Tools (Aug 2025)](https://www.nist.gov/news-events/news/2025/08/nist-releases-test-tools-accelerate-adoption-emerging-route-leak-mitigation)
- [BGP-SRx Reference Implementation](https://csrc.nist.gov/pubs/tn/2060/final)

### Academic Papers
- [NDSS 2025 — Securing BGP ASAP: ASPA and other Post-ROV Defenses](https://www.ndss-symposium.org/wp-content/uploads/2025-675-paper.pdf)
- [BGPsec Deployment Analysis (APNIC, May 2025)](https://blog.apnic.net/2025/05/23/bgpsec-could-you-run-it-if-you-wanted-to/)
- [GNN for BGP Anomaly Detection (ACM GNNet/CoNEXT)](https://dl.acm.org/doi/10.1145/3565473.3569188)
- [BGNN: BGP Anomaly Detection Using GNNs (HAL)](https://hal.science/hal-03688089v1/document)
- [Multi-View BGP Anomaly Detection via GAT (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S138912862200250X)
- [GNN Anomaly Detection Systematic Review (Springer, 2026)](https://link.springer.com/article/10.1007/s10462-026-11532-7)
- [Path Plausibility Algorithms in BGP (IEEE, 2024)](https://ieeexplore.ieee.org/document/10575088/)
- [Mitigating BGP Route Leaks (Höger, 2025)](https://onlinelibrary.wiley.com/doi/10.1002/nem.70002)

### Industry
- [ASPA: Next Layer of Routing Security (FastNetMon, Feb 2026)](https://fastnetmon.com/2026/02/25/aspa-the-next-layer-of-routing-security/)
- [ASPA: Cryptographic Upgrade for BGP Path Security (Noction)](https://www.noction.com/blog/aspa-rpki-cryptographic-upgrade-for-bgp-path-security)
- [Fixing BGP's Security Problems (The Register, Aug 2025)](https://www.theregister.com/2025/08/27/systems_approach_securing_internet_infrastructure/)

---

## 8. Summary: Why This Matters

| Gap in Current Project | How We Fix It |
|------------------------|---------------|
| No route leak attacks modeled | Route Leak Injector covers Type 1-3 leaks |
| No ASPA — the biggest BGP security development of 2025-2026 | Full ASPA validator with partial-deployment sweep |
| No OTC — the simplest real-world leak defense | OTC validator with adoption-rate modeling |
| Statistical anomaly detector ignores topology structure | GNN detector leverages graph structure for ≥15% improvement |
| No unified scoring — validators are islands | Path Plausibility engine fuses all signals probabilistically |
| Can't compare validators on route leaks | Enhanced evaluation includes leak detection metrics |

**Bottom line:** After this enhancement, Checkpoint Charlie will be the most comprehensive BGP path security simulation available — covering every major defense from 2017 (BGPsec) through 2026 (ASPA), from lightweight attribute checks (OTC) to deep learning (GNN), all evaluated head-to-head on a realistic topology with all known attack classes.
