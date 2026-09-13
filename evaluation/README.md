# KGRAG evaluation

This package evaluates hosted LLMs and the current local KGRAG workflow while
retaining the original Q1-Q6 task names.

| Group | File | Current purpose |
|---|---|---|
| Q1 | `Q1-Q6/Q1_Ontology.csv` | KGRAG ontology, vector fallback, applications, and kinetic KG concepts |
| Q2 | `Q1-Q6/Q2_MC_questions_fixed2.csv` | MOF single-attribute lookup |
| Q3 | `Q1-Q6/Q3_MC_questions_fixed.csv` | MOF multi-attribute filtering |
| Q4 | `Q1-Q6/Q4_chain_reasoning.csv` | Organic synthesis routes, multi-hop paths, and application-guided retrieval |
| Q5 | `Q1-Q6/Q5_conditional_reasoning.csv` | PINN/ODE kinetic parameters and recommended conditions |
| Q6 | `Q1-Q6/Q6_Multi-path_Comparison.csv` | Route comparison using yield, selectivity, activation energy, and measured support |

## Inputs

The benchmark CSVs are small and committed. Rebuilding them or running the
local workflow also requires these private/generated files:

- `data/nodes_with_context_mof.csv`
- `data/nodes_with_context_or_applications.csv`
- `data/kinetics_route_knowledge.json`
- `data/material_application_index.json`

Override their locations with `KGRAG_MOF_CONTEXT`, `KGRAG_OR_CONTEXT`,
`KGRAG_KINETICS_KNOWLEDGE`, and `KGRAG_APPLICATION_INDEX`.

## Collect answers

```bash
python evaluate_kgrag_pipeline.py \
  --dataset-dir Q1-Q6 \
  --output kgrag_model_answers.json

python evaluate_llms.py \
  --dataset-dir Q1-Q6 \
  --models "openai:gpt-5.5,qwen:qwen3.7-plus,deepseek:deepseek-v4-pro" \
  --output model_answers_updated.json
```

Hosted providers read `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, or
`DEEPSEEK_API_KEY` from the environment.

## Judge and aggregate

```bash
python judge_responses.py \
  --answers model_answers_updated.json \
  --judge-model openai:gpt-4.1-mini \
  --output llm_eval_results_updated.json

python collect_evaluations.py \
  --eval-root llm_eval_results_updated \
  --output evaluation_scores_updated.csv
```

Do not directly compare legacy score files with this benchmark: the question
set and local workflow were updated together.
