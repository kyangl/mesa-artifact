# MESA: Prioritizing Security-Critical Communication Channels in Multi-Agent Systems

MESA ranks inter-agent channels by expected security impact using topology and
clean-execution probes. It requires no attack outcomes from the target system.

This artifact contains the implementation, scenario and topology configs,
experiment drivers, analysis code, and compact aggregates used by the paper.

## Setup

Figure generation requires Python 3.9+:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Running experiments also requires [Ollama](https://ollama.com). The study uses
`llama3.1:8b`, `qwen3.5:9b`, `qwen3.5:27b`, `gemma4:e4b`, and `gemma4:26b`;
pull only the models needed for a selected run, then start `ollama serve`.

## Contents

```text
src/          MAS orchestration, attacks, defenses, features, and evaluation
config/       Scenario, topology, calibration, and task-fold definitions
runners/      Clean, attack, probe, stress-test, and defense experiments
analysis/     Feature, ranking, enforcement, and plotting code
data/         Compact aggregates used by the included figure scripts
```

The scenarios are customer service (20 tasks), software engineering (50
HumanEval tasks plus a separate difficulty-shift suite), and homogeneous
debate (55 GSM8K and 20 CommonsenseQA questions).

The configs include six named topologies and fixed-seed Erdős-Rényi and
Barabási-Albert graphs. Primary ranking and enforcement results use five named
topologies; mesh and random graphs are retained for breadth and stress tests.

Run commands from the repository root.

## Experiments

Examples:

```bash
# Clean and single-entry intervention sweeps
python runners/run_full_pilot.py --phase clean
python runners/run_full_pilot.py --phase attack

# Clean-execution feature probes
python runners/run_feature_probe.py --scenario config/scenarios/customer_service.yaml
python runners/run_dynamic_saliency.py --mode both

# NLI monitoring and matched control
python runners/run_cs_enforcement.py --model gemma4:e4b --topology sequential
python runners/run_cs_enforcement_control.py --model gemma4:e4b --topology sequential
```

Runners write raw JSON under `results/`. These runs are compute-intensive; use
`--max-tasks`, `--plan-only`, or other script-specific limits for smoke tests.

## Analysis

The following figures run from shipped aggregates and write to `figures/`:

```bash
python analysis/plots/fig_ranking_coverage_curve.py
python analysis/plots/fig_security_utility.py
python analysis/plots/fig_feature_selection.py
python analysis/plots/fig_langgraph_parity.py
```

Core aggregation scripts include:

```bash
python analysis/build_mas_features.py
python analysis/build_feature_matrix.py
python analysis/feature_selection_audit.py
python analysis/canonical_ranking_table.py
python analysis/canonical_ranking_curve.py
python analysis/control_adjusted_prevention.py
```

Aggregation requires raw results and fails explicitly when required inputs are
absent. `data/DATA_DICTIONARY.md` documents the shipped tables.

## Release boundary

Raw transcripts and generated figures are excluded because they are large and
may contain prompts or model output. They are not needed to inspect the code or
rebuild the four figures above.

When packaging an anonymous artifact, include the tracked project files but
exclude `.git/`, `results/`, `figures/`, virtual environments, caches, and OS
metadata. The tracked files contain no author names, affiliations, credentials,
or local absolute paths.
