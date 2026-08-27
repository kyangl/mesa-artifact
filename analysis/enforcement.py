"""Withhold flagged messages and measure the resulting outcome.

The frozen neutral placeholder avoids leaking threat information to receivers.
"""
import hashlib
import inspect
import numpy as np

NEUTRAL_PLACEHOLDER = ("[message withheld by monitoring policy]")
PLACEHOLDER_SHA16 = hashlib.sha256(
    NEUTRAL_PLACEHOLDER.encode()).hexdigest()[:16]

RANDOM_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)


def content_sha16(text):
    """Short content hash. Identifies a withheld message without storing it."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def cached_attack_payloads(messages, attacked_edge):
    """Return cached attack payloads on an edge in delivery order.

    Replaying them fixes the attack across baseline and defended runs.
    """
    if not attacked_edge:
        return []
    e = tuple(attacked_edge)
    return [m.get("attacked_content") for m in messages
            if (m.get("source"), m.get("target")) == e
            and m.get("was_attacked") and m.get("attacked_content")]


def random_ordering(edges, seed):
    """Deterministic random edge order. Seeds are committed, not drawn."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(edges))
    return [edges[i] for i in idx]


def random_orderings(edges, seeds=RANDOM_SEEDS):
    return [random_ordering(list(edges), s) for s in seeds]


def apply_withholding(messages, flagged_indices):
    """Replace flagged content while preserving source, target, and order."""
    flagged = set(flagged_indices)
    out = []
    for i, m in enumerate(messages):
        m2 = dict(m)
        if i in flagged:
            m2["content"] = NEUTRAL_PLACEHOLDER
            m2["withheld"] = True
        out.append(m2)
    return out


def enforcement_record(trial, flags, threshold, defended_success,
                       defended_utility=None, extra_calls=0, latency_s=0.0,
                       failure_reason=None, monitored=None, interventions=None,
                       provenance=None, extra_tokens=0, cache_key=None,
                       cache_hit=False):
    """Return an auditable row with baseline, defended, and pairing fields."""
    return {
        "task_id": trial.task_id,
        "attacked_edge": list(trial.attacked_edge) if trial.attacked_edge else None,
        "clean_correct": bool(trial.clean_correct),
        "original_success": bool(trial.success),
        "defended_success": (None if defended_success is None
                             else bool(defended_success)),
        "prevented": (None if defended_success is None
                      else bool(trial.success and not defended_success)),
        "flagged": [{"msg_index": i, "edge": list(e), "score": float(s)}
                    for i, e, s in flags],
        "threshold": float(threshold),
        "placeholder_sha16": PLACEHOLDER_SHA16,
        "final_utility": defended_utility,
        "extra_calls": int(extra_calls),
        "extra_tokens": int(extra_tokens),
        "latency_s": float(latency_s),
        "failure_reason": failure_reason,
        "monitored": ([list(e) for e in sorted(tuple(x) for x in monitored)]
                      if monitored is not None else None),
        "interventions": list(interventions or []),
        "n_interventions": sum(1 for i in (interventions or [])
                               if i.get("intervention_applied")),
        "provenance": dict(provenance or {}),
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit),
    }


def _supports(fn, name):
    """Whether ``fn`` accepts keyword ``name`` (directly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    p = params.get(name)
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY)


def make_rerun(execute_fn, det=None):
    """Build the rerun callback expected by ``evaluate_cell``.

    Downstream execution is regenerated after withholding; stored messages are
    provenance only. The detector may be bound here or supplied at call time.
    """
    bound_det = det

    def rerun(trial, monitored, threshold, det=None):
        d = det if det is not None else bound_det
        flags = d.flagged_messages(trial, monitored, threshold) if d else []
        if not flags:
            return trial
        withheld = apply_withholding(trial.messages, [i for i, _e, _s in flags])
        kw = {}
        for name, val in (("monitored", monitored), ("threshold", threshold),
                          ("flags", flags), ("det", d)):
            if _supports(execute_fn, name):
                kw[name] = val
        return execute_fn(trial, withheld, **kw)
    return rerun
