"""Workflow-aware edge features from clean calibration only.

F1 measures semantic non-recoverability from prior receiver context, F2 the
receiver response change under message masking, and F3 distance to a declared
decision sink. Missing probe measurements remain missing.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Predeclared sink roles per scenario (design doc section 5.2). Frozen.
SINKS = {
    "customer_service": ["supervisor", "transaction"],
    "software_engineering": ["supervisor"],
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_CACHE = {}


def _explicit_snapshot(repo_id, root=None):
    """Resolve a staged model to its snapshot directory.

    Cache-name resolution proved unreliable across container/library versions,
    so the model is loaded from an explicit directory when one is staged. This
    removes any dependency on how a particular huggingface_hub version lays
    out or looks up its cache.
    """
    root = Path(root or os.environ.get("HF_HOME", "")) / "hub"
    d = root / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return None
    snaps = sorted(p for p in d.iterdir() if p.is_dir())
    return str(snaps[0]) if snaps else None


def get_embedder(name=EMBED_MODEL):
    """Load the frozen sentence embedder once per process."""
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        local = _explicit_snapshot(name)
        _MODEL_CACHE[name] = SentenceTransformer(local or name)
    return _MODEL_CACHE[name]


# Persistent embedding cache. The login node enforces ulimit -t 7080s, and a
# process killed at that limit loses everything it computed. Caching by text
# hash means a re-run resumes instead of restarting, so the work completes
# across however many invocations it takes.
_CACHE_PATH = REPO / "results" / "embedding_cache.npz"
_EMB_CACHE = {}
_CACHE_DIRTY = False


def _cache_key(text, model_name):
    return hashlib.sha256(("%s|%s" % (model_name, text)).encode()).hexdigest()[:24]


def load_embedding_cache(path=None):
    global _EMB_CACHE
    path = Path(path or _CACHE_PATH)
    if path.exists() and not _EMB_CACHE:
        try:
            with np.load(path) as z:
                _EMB_CACHE = {k: z[k] for k in z.files}
        except Exception:
            _EMB_CACHE = {}
    return _EMB_CACHE


def save_embedding_cache(path=None):
    global _CACHE_DIRTY, _since_flush
    if not _CACHE_DIRTY:
        return
    path = Path(path or _CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy appends .npz when the name lacks it, so the temp name must
    # already end in .npz or the rename target will not exist.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **_EMB_CACHE)
    tmp.replace(path)
    _CACHE_DIRTY = False
    _since_flush = 0


# Flush every this many newly embedded texts. A finally block is not enough:
# SIGXCPU terminates the process outright, so cleanup code never runs and an
# end-of-run flush loses everything. Progress must be on disk before the kill.
FLUSH_EVERY = 200
_since_flush = 0


def _install_cpu_limit_handler():
    """Save and exit cleanly if the CPU limit fires, instead of core-dumping."""
    try:
        import signal

        def _on_xcpu(signum, frame):
            try:
                save_embedding_cache()
            finally:
                sys.exit(152)
        signal.signal(signal.SIGXCPU, _on_xcpu)
    except Exception:
        pass


def embed(texts, model_name=EMBED_MODEL, use_cache=True):
    global _CACHE_DIRTY, _since_flush
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=float)
    if not use_cache:
        model = get_embedder(model_name)
        return np.asarray(model.encode(texts, normalize_embeddings=True),
                          dtype=float)

    load_embedding_cache()
    keys = [_cache_key(t, model_name) for t in texts]
    missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts))
               if k not in _EMB_CACHE]
    if missing:
        model = get_embedder(model_name)
        vecs = model.encode([t for _, t in missing], normalize_embeddings=True)
        for (i, _), v in zip(missing, np.asarray(vecs, dtype=float)):
            _EMB_CACHE[keys[i]] = v
        _CACHE_DIRTY = True
        _since_flush += len(missing)
        if _since_flush >= FLUSH_EVERY:
            save_embedding_cache()
            _since_flush = 0
    return np.vstack([_EMB_CACHE[k] for k in keys])


def cosine_max(vec, matrix):
    """Max cosine similarity of vec against rows of matrix (both L2-normed)."""
    if matrix.shape[0] == 0:
        return None
    return float(np.max(matrix @ vec))


# ---------------------------------------------------------------- F3 (graph)

def sink_nodes(graph, topology_config, scenario_name):
    roles = SINKS.get(scenario_name, [])
    out = set()
    for cfg in topology_config.get("agents", []):
        if cfg.get("role") in roles and cfg["id"] in graph:
            out.add(cfg["id"])
    return out


def consequence_proximity(graph: nx.DiGraph, topology_config, scenario_name):
    """CP(e) = 1 / (1 + d(target(e), nearest sink))."""
    sinks = sink_nodes(graph, topology_config, scenario_name)
    out = {}
    for u, v in graph.edges():
        if not sinks:
            out[(u, v)] = float("nan")
            continue
        if v in sinks:
            d = 0
        else:
            dists = []
            for s in sinks:
                try:
                    dists.append(nx.shortest_path_length(graph, v, s))
                except nx.NetworkXNoPath:
                    continue
            d = min(dists) if dists else None
        out[(u, v)] = 1.0 / (1.0 + d) if d is not None else 0.0
    return out


# ------------------------------------------------------------- F1 (offline)

def semantic_non_recoverability(messages, model_name=EMBED_MODEL):
    """NR per edge from clean transcripts.

    ``messages`` is an ordered list of dicts with keys: step_idx, src, dst,
    content, and prior_context (the receiver's other available context at that
    step). NR = 1 when no alternative context existed -- the channel is the
    receiver's only source.
    """
    by_edge = {}
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        alternatives = [c for c in (m.get("prior_context") or []) if c.strip()]
        if not alternatives:
            nr = 1.0
        else:
            vecs = embed([content] + alternatives, model_name)
            nr = min(1.0, max(0.0, 1.0 - max(0.0, cosine_max(vecs[0], vecs[1:]))))
        by_edge.setdefault((m["src"], m["dst"]), []).append(nr)
    return {e: float(np.mean(v)) for e, v in by_edge.items()}


def receiver_context_before(messages, step_idx, receiver, role_context=None):
    """The receiver's other context strictly BEFORE this message.

    Enforces the F1 causality constraint: later transcript messages must never
    enter the comparison set.
    """
    prior = [m["content"] for m in messages
             if m["dst"] == receiver and m["step_idx"] < step_idx
             and (m.get("content") or "").strip()]
    if role_context:
        prior.append(role_context)
    return prior


# --------------------------------------------------------------- F2 (probe)

def receiver_response_sensitivity(probe_pairs, model_name=EMBED_MODEL):
    """RS per edge from (clean_output, probe_output) pairs.

    ``probe_pairs`` maps edge -> list of (clean_output, probe_output). Returns
    only edges that were actually probed; unmeasured edges are ABSENT rather
    than zero, because zero means "the receiver ignored this channel", which
    is a finding and not a placeholder.
    """
    out = {}
    for edge, pairs in probe_pairs.items():
        vals = []
        for clean_out, probe_out in pairs:
            if not (clean_out or "").strip() or not (probe_out or "").strip():
                continue
            v = embed([clean_out, probe_out], model_name)
            # clamp: identical outputs can give cos slightly > 1 in float
            vals.append(max(0.0, 1.0 - float(np.dot(v[0], v[1]))))
        if vals:
            out[edge] = float(np.mean(vals))
    return out
