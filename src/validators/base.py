"""Abstract base class that all validators implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from src.bgp_simulator import BGPRoute


class BaseValidator(ABC):
    """
    Every validator returns a 2-tuple:
      (is_valid: bool, processing_time_ms: float)

    is_valid = True  → the route passes validation (accepted)
    is_valid = False → the route is rejected / flagged as suspicious
    """

    name: str = "BaseValidator"

    @abstractmethod
    def validate(self, route: BGPRoute) -> Tuple[bool, float]:
        """Validate a single BGP route announcement."""
        ...
