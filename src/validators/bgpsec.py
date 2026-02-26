"""
BGPsec Validator
================
Full path validation: every consecutive hop pair is verified.
This is a wrapper around SelectiveHopValidator with positions=None.

Serves as the *gold standard* (100 % detection of topology-breaking attacks)
and the high-cost baseline for the security-vs-cost trade-off analysis.
"""

from __future__ import annotations

from src.topology import ASTopology
from src.validators.selective_hop import SelectiveHopValidator


class BGPsecValidator(SelectiveHopValidator):
    """
    Validates ALL hops in the AS_PATH.
    Timing: path_length × CRYPTO_VERIFY_MS
    """

    name = "BGPsec (Full Path)"

    def __init__(self, topology: ASTopology) -> None:
        super().__init__(topology, positions=None, label=self.name)
