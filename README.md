# MESA: Edge-Level Vulnerability in Multi-Agent LLM Systems

A novel method for modeling, quantifying, and predicting how **edge-level** vulnerabilities spread out in a multi-agent
LLM system (MAS) with static and dynamic features. 

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally for LLM inference

```bash
pip install -r requirements.txt

# Pull the models
ollama pull llama3.1:8b
ollama pull qwen3.5:9b
ollama serve            # in a separate terminal
```

## Layout

```
src/
├── topology/builder.py        Load YAML topologies
├── saliency/structural.py     Structural edge features → composite MESA score
├── agents/
│   ├── base_agent.py          Single Ollama-backed agent
│   ├── mas_runner.py          Orchestrator: 6 routing protocols + attack/defense hooks
│   └── langgraph_runner.py    Same interface, built on a LangGraph StateGraph
├── attacks/
│   ├── misinformation.py      Misinformation attacks
│   └── perturbation.py        Dynamic probes
└── evaluation/
    ├── metrics.py             Scenario evals (LLM-judge / exact-match)
    └── code_runner.py         Sandboxed HumanEval unit-test execution

config/
├── topologies/                6 designed graphs + random baselines
└── scenarios/                 Task definitions, ground truth, agent prompts

analysis/
├── cross_experiment_summary.py   Raw result JSONs → per-edge result table,
│                                 ASR, MESA ρ 
├── compute_structural_saliency.py Structural saliency report
├── gen_mesa_rho_table.py         LaTeX table: MESA ρ vs ASR
└── gen_appendix_tables.py        LaTeX tables: vulnerability concentration,
                                  clean-baseline accuracy
```

`src/` loads `config/` at runtime; `analysis/` imports `src/` and reads `config/`.
Run all commands from the repository root.

## Topologies & scenarios

**Topologies** (`config/topologies/`): `centralized` (hub), `sequential` (chain),
`hierarchical`, `decentralized` (ring), `mesh`, `hybrid`. 

**Scenarios** (`config/scenarios/`): `customer_service` (LLM-as-judge),
`software_engineering` (HumanEval unit tests), `homogeneous_debate`
(GSM8K / CommonsenseQA exact match).
