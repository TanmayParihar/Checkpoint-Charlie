"""
RPKI Validator
==============
Models Resource Public Key Infrastructure (RPKI) origin validation.

A Route Origin Authorization (ROA) maps a prefix to its authorised origin AS.
This validator checks only the *origin* of the route (as_path[-1]) against
the ROA database.  It cannot detect any form of path manipulation.

Computational model
-------------------
• One hash-table lookup per announcement plus one cryptographic verification
  if a ROA is found.
• Timing: HASH_LOOKUP_MS + CRYPTO_VERIFY_MS (if prefix is in ROA database).
• If the prefix has no ROA the route is considered "NotFound" and is accepted
  (this is the current real-world RPKI behaviour: unknown ≠ invalid).
"""

from __future__ import annotations

from typing import Tuple

import config
from src.bgp_simulator import BGPRoute
from src.topology import ASTopology
from src.validators.base import BaseValidator


class RPKIValidator(BaseValidator):
    """
    Validates the origin AS against a pre-built ROA database.

    Parameters
    ----------
    topology : ASTopology
        Provides the roa_db mapping.
    """

    name = "RPKI"

    def __init__(self, topology: ASTopology) -> None:
        self.roa_db = topology.roa_db   # prefix → authorised_origin_as

    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        """
        Returns
        -------
        (True, t)   → ROANotFound (no ROA, pass by default) or ROAValid
        (False, t)  → ROAInvalid (origin AS doesn't match ROA)
        """
        # Simulate hash-table lookup
        t = config.HASH_LOOKUP_MS

        authorised_origin = self.roa_db.get(route.prefix)
        if authorised_origin is None:
            # No ROA → route is "Unknown", accepted per RPKI policy
            return True, t

        # ROA exists → verify origin
        t += config.CRYPTO_VERIFY_MS
        claimed_origin = route.as_path[-1] if route.as_path else route.origin_as
        is_valid = (claimed_origin == authorised_origin)
        return is_valid, t
