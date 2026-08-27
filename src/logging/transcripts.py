"""Validated record schema for instrumented MESA runs.

Original, attacked, and enforced content remain separate so detector outcomes
can be reconstructed offline.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Field group -> the record keys that carry it.  "messages[]." means "present
# on every element of the messages list".
FIELD_GROUPS = {
    "configuration": ["scenario", "topology", "model", "task_id", "phase",
                      "seed", "policy", "budget_k", "threshold"],
    "model_version": ["run_context.model", "run_context.ollama_version",
                      "run_context.quantization", "run_context.detector_model",
                      "run_context.detector_revision"],
    "message_stage": ["messages[].step_idx", "messages[].src", "messages[].dst",
                      "messages[].edge_label", "messages[].stage"],
    "detector": ["messages[].detector_id", "messages[].detector_score",
                 "messages[].detector_threshold", "messages[].detector_flag",
                 "messages[].detector_latency_s", "messages[].detector_tokens"],
    "evidence": ["messages[].evidence"],
    "receiver_state": ["messages[].receiver_call", "messages[].probe_call"],
    "tool_call": ["messages[].tool_name", "messages[].tool_arguments",
                  "messages[].tool_valid", "messages[].tool_state_delta"],
    "outcome": ["final_output", "scores", "task_success", "unauthorized_action"],
    "cost": ["wall_time_s", "n_calls", "prompt_tokens", "completion_tokens"],
    "replay_lineage": ["parent_trial_id", "reused", "cache_key"],
    "hashes": ["run_context.git_commit", "run_context.container_digest",
               "run_context.model_digest", "run_context.config_digest"],
}

VALID_STAGES = ("clean", "attacked", "enforced")


def _git_commit() -> str:
    # Compute jobs may provide an untracked version stamp when .git is absent.
    stamp = Path(__file__).resolve().parents[2] / "config" / ".mesa_version"
    try:
        text = stamp.read_text().strip()
        if text:
            return text
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def config_digest(*paths: str) -> str:
    """Stable digest over the YAML configs a run depends on."""
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


@dataclass
class RunContext:
    model: str
    ollama_version: str
    quantization: str
    detector_model: Optional[str]
    detector_revision: Optional[str]
    git_commit: str
    container_digest: str
    model_digest: str
    config_digest: str

    @classmethod
    def capture(cls, model, ollama_version, quantization,
                detector_model=None, detector_revision=None,
                container_digest=None, model_digest=None,
                config_paths=()) -> "RunContext":
        return cls(
            model=model,
            ollama_version=ollama_version,
            quantization=quantization,
            detector_model=detector_model,
            detector_revision=detector_revision,
            git_commit=_git_commit(),
            container_digest=(container_digest
                              or os.environ.get("MESA_CONTAINER_DIGEST", "none")),
            model_digest=(model_digest
                          or os.environ.get("MESA_MODEL_DIGEST", "none")),
            config_digest=config_digest(*config_paths) if config_paths else "none",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReceiverCall:
    """Everything needed to replay one receiver invocation.

    ``messages`` is the exact list handed to the model, so an F2 probe can
    clone it and substitute only the delivered edge message.
    """
    agent_id: str
    role: str
    system: str
    messages: List[Dict[str, str]]
    response: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MessageRecord:
    step_idx: int
    src: str
    dst: str
    edge_label: str
    stage: str
    original_content: Optional[str]
    attacked_content: Optional[str]
    enforced_content: Optional[str]
    was_attacked: bool
    detector_id: Optional[str]
    detector_score: Optional[float]
    detector_threshold: Optional[float]
    detector_flag: bool
    detector_latency_s: Optional[float]
    detector_tokens: Optional[int]
    evidence: Optional[Any]
    receiver_call: Optional[ReceiverCall]
    tool_name: Optional[str]
    tool_arguments: Optional[Any]
    tool_valid: Optional[bool]
    tool_state_delta: Optional[Any]
    # F2 counterfactual: the same receiver call with only the delivered edge
    # message replaced by the fixed neutral placeholder.
    probe_call: Optional[ReceiverCall] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["receiver_call"] = (self.receiver_call.to_dict()
                              if self.receiver_call else None)
        d["probe_call"] = (self.probe_call.to_dict()
                           if self.probe_call else None)
        return d


@dataclass
class TrialRecord:
    run_context: RunContext
    scenario: str
    topology: str
    model: str
    task_id: str
    phase: str
    seed: int
    policy: str
    budget_k: int
    threshold: Optional[float]
    messages: List[MessageRecord]
    final_output: str
    scores: Dict[str, Any]
    task_success: bool
    unauthorized_action: bool
    wall_time_s: float
    n_calls: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    parent_trial_id: Optional[str]
    reused: bool
    cache_key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_context": self.run_context.to_dict(),
            "scenario": self.scenario,
            "topology": self.topology,
            "model": self.model,
            "task_id": self.task_id,
            "phase": self.phase,
            "seed": self.seed,
            "policy": self.policy,
            "budget_k": self.budget_k,
            "threshold": self.threshold,
            "messages": [m.to_dict() for m in self.messages],
            "final_output": self.final_output,
            "scores": self.scores,
            "task_success": self.task_success,
            "unauthorized_action": self.unauthorized_action,
            "wall_time_s": self.wall_time_s,
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "parent_trial_id": self.parent_trial_id,
            "reused": self.reused,
            "cache_key": self.cache_key,
        }


def _lookup(record: Dict[str, Any], dotted: str) -> Tuple[bool, Optional[str]]:
    """Return (found, problem_or_None) for one dotted/`[]` key path."""
    if dotted.startswith("messages[]."):
        leaf = dotted.split(".", 1)[1]
        msgs = record.get("messages")
        if not isinstance(msgs, list):
            return False, "messages: missing or not a list"
        for i, m in enumerate(msgs):
            if leaf not in m:
                return False, "messages[%d].%s: missing" % (i, leaf)
        return True, None
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, "%s: missing" % dotted
        node = node[part]
    return True, None


def validate_record(record: Dict[str, Any]) -> List[str]:
    """Return a list of problems; empty means the record is valid."""
    problems = []
    for group, keys in FIELD_GROUPS.items():
        for dotted in keys:
            ok, problem = _lookup(record, dotted)
            if not ok:
                problems.append("[%s] %s" % (group, problem))
    for i, m in enumerate(record.get("messages", []) or []):
        stage = m.get("stage")
        if stage not in VALID_STAGES:
            problems.append(
                "[message_stage] messages[%d].stage=%r not in %r"
                % (i, stage, list(VALID_STAGES)))
        if m.get("was_attacked") and not m.get("attacked_content"):
            problems.append(
                "[message_stage] messages[%d]: was_attacked but no "
                "attacked_content -- attack metadata must survive enforcement" % i)
    return problems


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
