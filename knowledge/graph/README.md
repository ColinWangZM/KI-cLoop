# Corpus graph construction

The builder links source records, canonical molecular identities, patent procedures, structural classes, and material properties into a graph for retrieval and knowledge reuse.

## Inputs and build

Install `knowledge/requirements-graph.txt`. Set these paths, or use their defaults:

| Variable | Input or output | Default |
| --- | --- | --- |
| `KICLOOP_GRAPH_WORK` | Graph output directory | `knowledge/graph/work/` |
| `KICLOOP_USPTO_DIR` | Source `output<year>.csv` files | `<graph work>/input/uspto/` |
| `KICLOOP_MATERIALS_ARCHIVE` | Materials ZIP archive | `<graph work>/input/materials.zip` |

USPTO CSV headers are `ID`, `source_xml_filename`, `procedure_text`, and `inchi`, `smiles`, and `name` fields prefixed by `reactant_`, `product_`, or `solvent_`. The materials archive contains record-property CSV tables and CIF files under `crystal_dataset/post_process_dataset/<dataset>/`, plus `statistics.json` and `README.txt`. The datasets are `oqmd`, `material_project`, `jarvis_data`, and `qmof`; their formula columns are `chemical_formula_reduced`, `formula_pretty`, `formula`, and `formula`, respectively. Source archives are supplied separately.

Run from the repository root:

```bash
python knowledge/graph/scripts/ingest_sources.py
python knowledge/graph/scripts/prepare_molecule_inputs.py
python knowledge/graph/scripts/check_domain_examples.py
python knowledge/graph/scripts/prepare_molecules.py
python knowledge/graph/scripts/prepare_materials.py
python knowledge/graph/scripts/build_graph.py
python knowledge/graph/scripts/validate_graph.py
```

Outputs include graph Parquet files and validation reports covering identifiers, relations, source coverage, and provenance. `check_domain_examples.py` also runs independently of the source corpora.

## Physical quantities

The property graph associates values with units, species, reaction models, and source contexts, and connects equations to their inputs and assumptions.

Set `KICLOOP_PROPERTY_WORK` to a directory containing `inputs/closed_loop_models.json` and `inputs/kinetic_parameter_summary.csv`. The model JSON is exported by `python knowledge/build_kinetic_graph.py`; parameters come from the kinetics module. The commands below use the base graph at `KICLOOP_GRAPH_WORK` and generate `inputs/project_species.json`, property records, and validation reports.

```bash
python knowledge/graph/properties/prepare_species.py
python knowledge/graph/properties/build_properties.py
python knowledge/graph/properties/validate_properties.py
```
