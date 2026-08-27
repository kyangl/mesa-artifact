"""Cache one perturbation per scenario-model-topology-task-role key.

Reusing generated noise across edges, folds, and reruns isolates edge effects
and makes resumed jobs reproducible. Fixed ablation text needs no cache.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable, Dict, Optional


class PerturbationCache:
    """Generate-once, reuse-everywhere noise text keyed by task and role."""

    def __init__(self, path=None, generator: Optional[Callable] = None,
                 scenario="", model="", topology=""):
        self.path = Path(path) if path else None
        self.generator = generator
        self.scenario = scenario
        self.model = model
        self.topology = topology
        self._lock = threading.Lock()
        self._store: Dict[str, str] = {}
        if self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text())
            except Exception:
                self._store = {}

    def key(self, task_id, sender_role):
        return "|".join([self.scenario, self.model, self.topology,
                         str(task_id), str(sender_role)])

    def get(self, task_id, sender_role, message=""):
        """Return the cached perturbation, generating it once if absent."""
        k = self.key(task_id, sender_role)
        with self._lock:
            if k in self._store:
                return self._store[k]
        text = self.generator(message, sender_role) if self.generator else ""
        with self._lock:
            # Another caller may have filled it while the LLM call was in
            # flight; keep the first value so the key stays single-valued.
            if k not in self._store:
                self._store[k] = text
                self._flush()
            return self._store[k]

    def _flush(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._store, indent=2))
        tmp.replace(self.path)          # atomic; concurrent jobs never tear

    def digest(self):
        """Stable digest of the cache contents, for provenance."""
        blob = json.dumps(self._store, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def __len__(self):
        return len(self._store)


def pair_key(scenario, model, topology, task_id):
    """The exact identity a clean and an attacked run must share to be paired.

    Pairing on anything coarser -- topology-level mean accuracy, say -- lets a
    clean result from one task offset an attacked result from another, which
    silently mixes task difficulty into the measured delta.
    """
    return (scenario, model, topology, task_id)


def pair_runs(clean_records, attacked_records, edge=None):
    """Pair clean and attacked trials on exact identity.

    Returns (pairs, unmatched_clean, unmatched_attacked) so missing cells are
    reported rather than dropped silently.
    """
    def key(r):
        return pair_key(r.get("scenario"), r.get("model"), r.get("topology"),
                        r.get("task_id"))

    clean_by = {key(r): r for r in clean_records if r.get("task_id")}
    pairs, unmatched_att = [], []
    for a in attacked_records:
        if not a.get("task_id"):
            continue
        if edge is not None:
            ae = a.get("attack_edge")
            if not ae or tuple(ae) != tuple(edge):
                continue
        k = key(a)
        if k in clean_by:
            pairs.append((clean_by[k], a))
        else:
            unmatched_att.append(k)
    matched = {key(c) for c, _ in pairs}
    unmatched_clean = [k for k in clean_by if k not in matched]
    return pairs, unmatched_clean, unmatched_att
