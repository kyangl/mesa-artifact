"""NLI evidence-checking detector for monitored inter-agent messages.

Lives under src/ so it ships to compute nodes alongside ``config`` and
``runners``. Runs on CPU: the checker is a base-sized encoder.

It targets *misinformation* -- a claim that contradicts authoritative evidence
-- so the customer-service defense is entailment against the policy and
transaction records rather than an injection classifier.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.defenses.base import Detector, DetectorVerdict

NLI_MODEL = "microsoft/deberta-base-mnli"


class NLIEvidenceDetector(Detector):
    """Flags a message that CONTRADICTS the authoritative evidence.

    Misinformation is not an injection string; it is a claim inconsistent with
    the record. So the signal is entailment-model contradiction probability
    against the policy/transaction evidence for that edge, not a classifier
    trained on jailbreak text.

    An edge carrying no meaningful evidence is not scored -- returning a
    confident "benign" there would silently inflate the true-negative rate.
    """

    detector_id = "nli_evidence_contradiction"

    def __init__(self, threshold: float = 0.5, model_name: str = NLI_MODEL,
                 device: str = "cpu", max_length: int = 512):
        super().__init__(threshold=threshold)
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tok = None
        self._model = None
        self._contradiction_idx = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self._torch = torch
        from src.saliency.mas_features import _explicit_snapshot
        src_path = _explicit_snapshot(self.model_name) or self.model_name
        self._tok = AutoTokenizer.from_pretrained(src_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            src_path).to(self.device).eval()
        labels = getattr(self._model.config, "id2label", {}) or {}
        for i, name in labels.items():
            if str(name).upper().startswith("CONTRAD"):
                self._contradiction_idx = int(i)
        if self._contradiction_idx is None:
            self._contradiction_idx = 0

    @property
    def revision(self):
        return self.model_name

    def score(self, message, edge, local_context=None, evidence=None):
        if not evidence or not str(evidence).strip():
            # Not applicable: report abstention rather than a benign verdict.
            return DetectorVerdict(flag=False, confidence=float("nan"),
                                   latency_s=0.0, token_cost=0)
        self._load()
        t0 = time.time()
        enc = self._tok(str(evidence), message or "", return_tensors="pt",
                        truncation=True, max_length=self.max_length).to(self.device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        confidence = float(probs[self._contradiction_idx].item())
        return DetectorVerdict(
            flag=confidence >= self.threshold,
            confidence=confidence,
            latency_s=time.time() - t0,
            token_cost=int(enc["input_ids"].shape[-1]),
        )


def sweep_threshold(scores: Sequence[float], labels: Sequence[int],
                    grid=None):
    """TPR/FPR across thresholds. Labels: 1 = should be flagged.

    NaN scores are abstentions and are excluded from both rates rather than
    counted as correct negatives.
    """
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 51)
    rows = []
    for t in grid:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        rows.append({
            "threshold": float(t),
            "tpr": tp / max(1, tp + fn),
            "fpr": fp / max(1, fp + tn),
            "precision": tp / max(1, tp + fp),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return {"n_scored": int(len(s)), "n_abstained": int((~ok).sum()),
            "curve": rows}
