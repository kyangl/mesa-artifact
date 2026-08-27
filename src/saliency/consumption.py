"""Map delivered messages to the receiver calls that consume them.

The map preserves co-consumed context for F1 and repeated edge occurrences for
F2, including aggregation and multi-round protocols.
"""

import re
from typing import Any, Dict, List, Optional


def extract_role_evidence(system_prompt: str) -> Optional[str]:
    """The mock-data block an agent carries in its system prompt, if any."""
    if not system_prompt:
        return None
    m = re.search(r"Available data:\s*```json\s*(.*?)```", system_prompt, re.S)
    return m.group(1).strip() if m else None


def build_consumption_map(edge_log, agents):
    """Which receiver call consumed each delivered message occurrence.

    Returns a list of occurrence dicts, one per edge_log entry, each with:
      step_idx, src, dst, content, call_index (or None if never consumed).

    A message is consumed by a call when its content appears in that call's
    final user message. One call may consume many messages (aggregation) and
    one edge may appear in many calls (multiple rounds).
    """
    occurrences = []
    # Track, per receiver, which call each occurrence was assigned to so that
    # repeated deliveries map to successive calls rather than all to the first.
    assigned = {}
    for step_idx, m in enumerate(edge_log):
        dst = m["target"]
        content = (m.get("content") or "").strip()
        call_index = None
        agent = agents.get(dst)
        calls = list(getattr(agent, "call_log", [])) if agent else []
        if content:
            start = assigned.get((dst, content), 0)
            for i in range(start, len(calls)):
                last = calls[i].messages[-1].get("content", "") if calls[i].messages else ""
                if content in last:
                    call_index = i
                    assigned[(dst, content)] = i + 1
                    break
            if call_index is None:
                # Fall back to any call that contains it (repeat delivery of
                # identical content already consumed).
                for i, c in enumerate(calls):
                    last = c.messages[-1].get("content", "") if c.messages else ""
                    if content in last:
                        call_index = i
                        break
        occurrences.append({
            "step_idx": step_idx,
            "src": m["source"],
            "dst": dst,
            "content": content,
            "call_index": call_index,
        })
    return occurrences


def comparison_context(occurrences, occ, agents):
    """Everything else the receiver had when it consumed this message.

    Co-consumed messages from the same call, messages consumed in strictly
    earlier calls, and the receiver's role/local evidence.
    """
    dst, call_index = occ["dst"], occ["call_index"]
    context = []

    for other in occurrences:
        if other is occ or other["dst"] != dst or not other["content"]:
            continue
        if other["call_index"] is None:
            continue
        if call_index is None:
            # Unconsumed message: fall back to strictly-earlier deliveries.
            if other["step_idx"] < occ["step_idx"]:
                context.append(other["content"])
        elif other["call_index"] < call_index:
            context.append(other["content"])          # earlier call
        elif other["call_index"] == call_index:
            context.append(other["content"])          # co-consumed
    agent = agents.get(dst)
    if agent is not None:
        calls = list(getattr(agent, "call_log", []))
        idx = call_index if call_index is not None else (len(calls) - 1)
        if 0 <= idx < len(calls):
            evidence = extract_role_evidence(calls[idx].system)
            if evidence:
                context.append(evidence)
    return context
