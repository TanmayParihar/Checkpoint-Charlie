"""
Global configuration for the BGP routing security simulation.
"""

# ── Topology parameters ──────────────────────────────────────────────────────
NUM_ASES        = 2000   # Total number of Autonomous Systems
TIER1_COUNT     = 15     # Tier-1 backbone ASes (fully-meshed peers)
TIER2_COUNT     = 300    # Tier-2 regional ISPs
# Tier-3 edge networks make up the remainder

RANDOM_SEED     = 42

# ── BGP simulation parameters ────────────────────────────────────────────────
NUM_ROUTES          = 20_000   # Total route announcements generated
RPKI_COVERAGE       = 0.45     # Fraction of prefixes covered by ROAs
ATTACK_RATE         = 0.15     # Fraction of announcements that are attacks

# Distribution of attack types (must sum to 1.0)
ATTACK_TYPE_DIST = {
    "origin_hijack":    0.40,
    "path_shortening":  0.30,
    "path_fabrication": 0.30,
}

# ── Timing constants (milliseconds per operation) ───────────────────────────
# NOTE: these are *model parameters* drawn from published BGPsec benchmarks,
# not wall-clock measurements.  All timing comparisons are therefore relative
# (validator A vs. validator B), not absolute deployment estimates.
CRYPTO_VERIFY_MS    = 0.50   # RSA-2048 signature verification per hop (model)
HASH_LOOKUP_MS      = 0.01   # Hash-table / dict lookup (model)

# ── Selective-hop combinations to benchmark ──────────────────────────────────
# Position 0  = origin AS (rightmost in AS_PATH)
# Position 1  = AS immediately after origin
# Position -1 = receiving peer (leftmost in AS_PATH)
# None        = verify all hops (BGPsec)
#
# NOTE on "Selective: Origin Only" vs. the RPKIValidator class:
#   RPKIValidator   – models real-world RPKI: checks prefix→origin binding in
#                     the ROA database.  Only ~45 % of prefixes have a ROA, so
#                     detection is limited by deployment coverage.
#   "Selective: Origin Only" (SelectiveHopValidator, positions=[0]) – checks
#                     topological adjacency at the origin hop for *every* route,
#                     regardless of ROA coverage.  It represents the theoretical
#                     ceiling of origin-only cryptographic verification at 100 %
#                     deployment and is a different mechanism entirely.
HOP_COMBINATIONS = {
    "origin_only":         {"positions": [0],         "label": "Selective: Origin Only"},
    "origin_first":        {"positions": [0, 1],      "label": "Origin + 1st Hop"},
    "origin_last":         {"positions": [0, -1],     "label": "Origin + Last Hop"},
    "origin_first_last":   {"positions": [0, 1, -1],  "label": "Origin + 1st + Last"},
    "origin_first2_last":  {"positions": [0, 1, 2, -1], "label": "Origin + 2 Hops + Last"},
    "full_path":           {"positions": None,         "label": "BGPsec (Full Path)"},
}

# ── Anomaly-detection thresholds ─────────────────────────────────────────────
ANOMALY_SCORE_THRESHOLD     = 0.5   # Routes above this score are flagged
PATH_LENGTH_ZSCORE_LIMIT    = 2.5   # Z-score above which path length is suspicious
NEW_AS_PENALTY              = 0.25  # Score bump for each never-seen-before AS
ORIGIN_CHANGE_PENALTY       = 0.45  # Score bump if origin AS changes for a prefix
VALLEY_FREE_PENALTY         = 0.30  # Score bump for valley-free violation

# ── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
