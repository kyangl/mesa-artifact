"""Common interface for MESA monitor detectors.

A detector scores one message on one monitored edge and returns a verdict.
Enforcement is quarantine: the message is replaced with a fixed notice.  A
detector never restores the clean message -- that capability belongs only to
the oracle upper bound, which is a separate code path
(``EdgeMessage.oracle_revert``).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass
class DetectorVerdict:
    flag: bool
    confidence: float
    latency_s: float
    token_cost: int


class Detector(ABC):
    detector_id = "abstract"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def score(self, message: str, edge: Tuple[str, str],
              local_context: Optional[Any] = None,
              evidence: Optional[Any] = None) -> DetectorVerdict:
        """Score one message on one monitored edge."""


class StubDetector(Detector):
    """Deterministic keyword detector.

    Exists so the monitor/quarantine path and the record schema can be
    validated before real detector weights are staged.  Never used for
    reported results.
    """

    detector_id = "stub_keyword_v1"

    def __init__(self, keywords: Sequence[str], threshold: float = 0.5):
        super().__init__(threshold=threshold)
        self.keywords = [k.lower() for k in keywords]

    def score(self, message, edge, local_context=None, evidence=None):
        t0 = time.time()
        low = (message or "").lower()
        hits = sum(1 for k in self.keywords if k in low)
        # Graded, so a threshold actually discriminates: the fraction of
        # configured keywords the message matches.
        confidence = hits / float(len(self.keywords)) if self.keywords else 0.0
        return DetectorVerdict(
            flag=confidence >= self.threshold,
            confidence=confidence,
            latency_s=time.time() - t0,
            token_cost=len((message or "").split()),
        )
