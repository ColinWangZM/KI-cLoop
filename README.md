# KI-cLoop

Code and research data for **Closing the reaction-to-process loop with knowledge-intensified AI**.

KI-cLoop combines knowledge retrieval, automated reaction experiments, hybrid PINN–ODE kinetic inference, condition screening and environmental accounting. The Paal–Knorr reaction provides the model-assessment benchmark; four peroxide routes provide the application datasets.

## Contents

| Directory | Contents |
| --- | --- |
| `paal_knorr/` | Processed concentration records, stoichiometric calculations, kinetic inference and grouped validation |
| `kinetics/` | Four-route processed concentration inputs, kinetic models, uncertainty and sensitivity analysis |
| `analysis/` | Condition screening, environmental accounting and thirteen knowledge-reuse cases |
| `knowledge/` | Graph construction, retrieval, questions and scientific reference answers, and reaction-space visualization |

Computational reproduction starts from the supplied processed concentration records. Each experimental trajectory contains an inlet state and nonzero-time observations. Fitting-error metrics exclude inlet states. The module descriptions distinguish measured and reconstructed concentrations. Experimental sampling and HPLC quantification are described in the manuscript and Supplementary Information.

## Install and reproduce

Use Python 3.10 on Linux or WSL with a C++17 compiler. Package versions are specified in `requirements.txt`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python verify.py --plots --screening
```

`verify.py` checks file hashes, CSV structure, stoichiometric concentration calculations, fitted ODE predictions, bootstrap intervals, environmental accounting and numerical case calculations. It also verifies application and synthesis-route retrieval with its source records, and constructs and queries the project graph. The optional flags render plots and repeat the finite condition search. Generated files are written to `outputs` directories.

| Check | Expected result |
| --- | --- |
| Paal–Knorr apparent activation energy | 12.3271 kJ/mol |
| Product R² for Routes 1–4 | 0.986274, 0.988612, 0.991984, 0.927909 |
| Screened GWP for Routes 1–4 | 5.074584, 3.268239, 3.467700, 12.302637 kg CO2-eq/kg product |

To fit the kinetic models:

```bash
python paal_knorr/run.py --mode all
python kinetics/reproduce.py --mode full
```

Training proceeds through ODE initialization, PINN pretraining and joint training, then final ODE parameter refinement. Uncertainty and predictive checks use 200 condition-level residual bootstrap refits and grouped validation of the ODE estimator. See the module READMEs for individual run modes and outputs.

## Knowledge and case records

The question-answer dataset contains 120 questions and scientific reference answers. [`knowledge/benchmark/references.jsonl`](knowledge/benchmark/references.jsonl) provides concise answers; [`reference_answers.jsonl`](knowledge/benchmark/reference_answers.jsonl) includes question text, calculation steps, numerical facts and source records, using the reported final PINN–ODE parameters and corresponding concentration data. See the [dataset description](knowledge/README.md#question-answer-data) for the file contents.

Model response logs, individual assessment records and scripts for generating or grading responses are not included in this distribution. The model-comparison scores are reported in the manuscript and Supplementary Information.

The thirteen knowledge-reuse cases provide questions, analyses and supporting evidence, including analytical thermal estimates. A separate graph record describes the COMSOL reactor geometry and coupled physical fields.

The project graph is generated from the experimental and kinetic records. See [knowledge/README.md](knowledge/README.md) for retrieval and reference-answer data, and [knowledge/graph/README.md](knowledge/graph/README.md) for corpus-scale graph construction using external datasets.

The supplied material, alias, application-label and synthesis-route index, together with its supporting source records, forms a task-specific subset of the knowledge graph for reproducing the application and synthesis-route retrieval stage of RAG. The complete candidate index within this subset is included. This retrieval stage runs with the Python standard library and returns source-linked records.

```bash
python knowledge/query.py "Find synthesis routes for materials used as oxidizers or initiators."
python knowledge/query.py "What are the synthesis routes for TBHP and TBPB?"
python knowledge/verify_application_retrieval.py
```

## Manuscript correspondence

| Supplementary Information | Repository material |
| --- | --- |
| S1–S2: sources, extraction, retrieval and question-answer data | `knowledge/` |
| S3.1–S3.5 and S5: peroxide experimental conditions and processed concentration records | `kinetics/data/` |
| S3.6 and S4: Paal–Knorr experiments and model assessment | `paal_knorr/` |
| S6: peroxide kinetics and condition screening | `kinetics/results/`, `analysis/screen.py` |
| S7.1–S7.2: environmental accounting | `analysis/lca.py`, `analysis/reference/` |
| S7.3–S7.4: reactor representation | `knowledge/graph/reactor_evidence.jsonl` |
| S8: knowledge reuse | `analysis/data/cases/`, `knowledge/build_kinetic_graph.py` |

## Supplementary Data correspondence

| Supplementary Data | Repository |
| --- | --- |
| `peroxide/*_measurements.csv` | `kinetics/data/` with the same filenames |
| `paal_knorr/chromatography.csv` | `paal_knorr/data/chromatography.csv` |
| `question_answer_evaluation/*.jsonl` | `knowledge/benchmark/` with the same filenames |
| `knowledge_reuse_cases.json` | The thirteen JSON records in `analysis/data/cases/`, ordered by case number |

Shared CSV and JSONL files are identical in both packages. See `LICENSE` and `NOTICE` for licensing and attribution.
