"""Frozen task folds for directional cross-fitting.

Workload-derived features from one fold score outcomes on the other. Customer
service is stratified by decision class; ordered benchmarks are interleaved to
avoid contiguous difficulty splits. Graph-only features require no split.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO / "config" / "task_folds.json"

FOLD_A = "A"
FOLD_B = "B"


def _cs_stratum(task) -> str:
    """Binary decision class of a customer-service task."""
    gt = task.get("ground_truth") or {}
    if isinstance(gt, dict):
        for key in ("return_eligible", "refund_eligible"):
            if key in gt:
                return "eligible" if gt[key] else "ineligible"
    return "unknown"


def stratify(tasks, scenario_name) -> Dict[str, str]:
    """Assign each task id to fold A or B, deterministically.

    Within each stratum tasks are interleaved in their listed order, so an
    ordering that encodes difficulty is split evenly rather than halved.
    """
    if scenario_name == "customer_service":
        keyed = [(_cs_stratum(t), t["id"]) for t in tasks]
    else:
        keyed = [("all", t["id"]) for t in tasks]

    by_stratum = {}
    for stratum, tid in keyed:
        by_stratum.setdefault(stratum, []).append(tid)

    assignment = {}
    for stratum in sorted(by_stratum):
        for i, tid in enumerate(by_stratum[stratum]):
            assignment[tid] = FOLD_A if i % 2 == 0 else FOLD_B
    return assignment


def freeze(scenarios: Dict[str, list], path=FROZEN_PATH) -> Dict:
    """Compute and persist the split. Refuses to silently change an existing one."""
    folds = {name: stratify(tasks, name) for name, tasks in scenarios.items()}
    payload = {
        "note": ("Frozen two-fold cross-fitting split. Fold A features pair "
                 "with fold B outcomes and vice versa. Stratified on the "
                 "decision class where one exists, interleaved otherwise, "
                 "because task order encodes difficulty."),
        "folds": folds,
    }
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        # Compare only requested scenarios; the frozen file is additive.
        current = existing.get("folds", {})
        disagreeing = [name for name, split in folds.items()
                       if name in current and current[name] != split]
        if disagreeing:
            raise RuntimeError(
                "frozen task split at %s disagrees with a freshly computed "
                "one for %s; refusing to overwrite. Delete it deliberately if "
                "the scenario task list genuinely changed."
                % (path, ", ".join(sorted(disagreeing))))
        missing = {n: s for n, s in folds.items() if n not in current}
        if missing:
            current.update(missing)
            existing["folds"] = current
            path.write_text(json.dumps(existing, indent=2))
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load(path=FROZEN_PATH) -> Dict[str, Dict[str, str]]:
    return json.loads(Path(path).read_text())["folds"]


def fold_of(scenario_name, task_id, folds=None) -> str:
    folds = folds if folds is not None else load()
    return folds.get(scenario_name, {}).get(task_id)


def opposite(fold: str) -> str:
    return FOLD_B if fold == FOLD_A else FOLD_A


def split_ids(scenario_name, folds=None) -> Tuple[List[str], List[str]]:
    folds = folds if folds is not None else load()
    m = folds.get(scenario_name, {})
    a = sorted([t for t, f in m.items() if f == FOLD_A])
    b = sorted([t for t, f in m.items() if f == FOLD_B])
    return a, b
