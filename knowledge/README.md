# Knowledge graph, retrieval and reference answers

This module supports graph construction, evidence retrieval, kinetic-knowledge export, literature extraction, scientific reference-answer data and reaction-space visualization.

| Entry point | Inputs |
| --- | --- |
| `benchmark/` | Questions and scientific reference answers |
| `build_kinetic_graph.py` | Experimental data, fitted kinetics and knowledge-reuse cases |
| `query.py` | A question; application and route index, or project SQLite graph for kinetic questions |
| `application_retrieval.py` | Material, alias, application and synthesis-route index with source records |
| `verify_application_retrieval.py` | Index integrity, source correspondence and retrieval behavior |
| `graph/scripts/` | External USPTO and materials datasets |
| `graph/properties/` | Base graph, kinetic models and property inputs |
| `extraction/protocol.md` | Document evidence, ontology and extraction schema |
| `reaction_space/plot.py` | Saved coordinates, reaction classes and support labels |

## Run

From the repository root:

```bash
python knowledge/build_kinetic_graph.py
python knowledge/query.py "Find synthesis routes for materials used as oxidizers or initiators."
python knowledge/query.py "What are the synthesis routes for TBHP and TBPB?"
python knowledge/query.py "For Path 1, what is the main rate multiplier from 40 to 60 °C at fixed composition?"
python knowledge/query.py "Describe the reactor model and its coupled physical fields."
python knowledge/reaction_space/plot.py
```

These commands build and query the project graph and render reaction-space panels. Outputs are saved under `knowledge/outputs/` or `knowledge/reaction_space/outputs/`.

The project graph links experimental conditions, concentration trajectories and fitted kinetics. It also includes a qualitative reactor-model description. Kinetic queries should name a route (`Path 1`–`Path 4` or `Route 1`–`Route 4`). Retrieved records support calculations of temperature effects, reaction-order sensitivities and endpoint metrics.

## Application and synthesis-route retrieval

The material, alias, application-label and synthesis-route records supplied here form a task-specific subset of the knowledge graph for reproducing the application and synthesis-route retrieval stage of RAG. `data/material_application_index.json` contains the complete candidate index within this subset. `data/application_route_sources.jsonl` supplies the corresponding source records, linked by `source_id`.

Application questions match candidate labels such as oxidant and initiator; multiple requested applications use an OR match. Direct material questions match chemical names and aliases, including TBHP and TBPB. Candidates with registered experimental routes are ranked first, followed by route count and material name. The default response contains up to six materials and five routes per material, with the matching and returned counts reported separately.

Application labels are assigned by name rules and curated aliases. Source records distinguish extracted reaction contexts from registered experimental routes. These records support inspection of the retrieved candidates and their origins; candidate selection requires chemical assessment. The four experimental routes reference the distributed experimental and kinetic records.

The index contains 2,303 materials and 2,310 route records. Among the source records, 2,300 extracted contexts carry patent identifiers, six have no recorded external identifier, and four describe project experiments. The source fields retain these distinctions. A record with a documented conflict between its procedure and assigned chemical roles remains available for inspection in the index and source file and is excluded from returned synthesis routes.

This retrieval stage formats evidence deterministically using the Python standard library. The application queries above run before project-graph construction and do not require model inference. The query result includes the source records for the returned routes. Kinetic and reactor questions continue to use the project SQLite graph.

```bash
python knowledge/verify_application_retrieval.py
python knowledge/verify_application_retrieval.py --graph knowledge/outputs/kinetic_graph/graph.sqlite
```

These checks cover all index-to-source links, application synonyms, direct material queries, chemical identities in the four experimental routes, unknown queries and query routing. The second command also queries the constructed SQLite graph to verify the kinetic-query entry point. Both stages are included in the repository-level `verify.py` workflow.

## Question-answer data

The dataset contains 120 questions across six categories. `benchmark/questions.jsonl` provides the question text and category information. `benchmark/references.jsonl` contains concise scientific reference answers. `benchmark/reference_answers.jsonl` provides the question text, answers, calculation steps, numerical facts and source records; kinetic calculations use the reported final PINN–ODE parameters and the corresponding concentration data.

These files are scientific reference material. Model response logs, individual assessment records and scripts for generating or grading responses are not included in this distribution. The model-comparison scores are reported in the manuscript and Supplementary Information.

## Corpus graph construction

Install `knowledge/requirements-graph.txt` and follow [graph/README.md](graph/README.md) to prepare the external source datasets, construct the graph and validate its records and relations.
