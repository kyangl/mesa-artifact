# Data dictionary

This directory contains compact aggregates, not transcripts, embeddings,
detector logits, or trial-level outputs.

## Tidy tables

| File | Contents |
|---|---|
| `feature_selection.csv` | Paired coverage-AUC contrasts for the feature freeze |
| `ranking_coverage.csv` | MESA, random, and oracle coverage at 10% and 20% budgets |
| `ranking_coverage_curve.csv` | Coverage over the 10-100% budget sweep |
| `security_utility.csv` | Control-adjusted prevention, raw prevention, and clean utility |
| `model_breadth.csv` | Corrected attack success rate by open-weight model |
| `langgraph_per_edge_asr.csv` | Per-edge ASR for the native and LangGraph runners |

The JSON files preserve the corresponding structured results and diagnostics.

## Common columns

| Column | Meaning |
|---|---|
| `metric`, `estimate` | Quantity and value; an empty estimate means not measured |
| `ci_lo`, `ci_hi` | Interval bounds when available |
| `denominator`, `unit` | Sample size and unit of analysis |
| `scope` | Configurations included in the estimate |
| `model`, `domain`, `stratum` | Reported slice |
| `budget` | Monitoring budget or target false-positive rate |
| `fold_direction` | Cross-fit directions included in the estimate |
| `status` | `canonical` paper result or `diagnostic` supporting result |
| `feature_block` | Score used for ranking or enforcement |
| `source`, `source_sha16` | Upstream artifact and short content hash |

## Interpretation

- Missing estimates are not zeros.
- Domains and strata are not pooled unless `scope` says otherwise.
- Exact random coverage is `ceil(budget * |E|) / |E|`.
- Only the predeclared 10% and 20% budgets carry paired intervals in the
  coverage curve; other budgets describe its shape.
