# Condition screening and knowledge reuse

These programs use the four-route hybrid PINN–ODE estimates in `../kinetics/results` to screen reaction conditions, calculate screening life-cycle inventories, and assemble thirteen knowledge-reuse records. Concentrations are in mol/L and residence times in seconds unless otherwise specified.

## Reproduce

Run from the repository root after installing the core requirements:

```bash
python analysis/lca.py --verify
python analysis/cases.py
python analysis/draw_results.py
```

`lca.py` calculates the inventory from the selected conditions and verifies GWP and route scores against the reference results. Expected GWP values for Routes 1–4 are 5.074584, 3.268239, 3.467700, and 12.302637 kg CO2-eq per kg target product.

To rerun the finite condition search and then calculate its inventory:

```bash
python analysis/screen.py
python analysis/lca.py --conditions outputs/analysis/screening/screened_optimal_conditions.csv
```

The search combines observed feed ratios and catalyst levels within stock-solution mixing constraints and measured concentration bounds. It uses a 1 °C temperature grid and a residence-time search with local refinement. Results are model-selected conditions, with parameter-bootstrap intervals evaluated at those conditions. The Route 1 predicted yield is 51.7408%, compared with 43.9207% at the measured point selected by the same objective, a relative model-predicted increase of about 17.8%.

## Inventory assumptions

The functional unit is 1 kg target product. The inventory assumes 95% recovery of unreacted feed, 5% catalyst makeup, and substitution credits for DTBP or tert-butanol coproducts. Route 4 includes acetonitrile from its 38.88 wt% TBAI solution. Process energy is excluded.

`data/lca_factors.csv` records the factors and their origins, including inferred balances, proxies and EchoChemAI predictions. A sensitivity analysis varies benzaldehyde and acetonitrile factors by ±50%. Route scoring weights are 0.40 for yield, 0.25 for selectivity, 0.15 for productivity, 0.10 for lower apparent activation energy and 0.10 for lower estimated GWP.

## Thirteen knowledge-reuse records

`data/cases` contains thirteen records covering kinetic interpretation, condition screening, literature-based transfer scenarios and thermal estimates. `cases.py` recomputes kinetic scenarios and temperature-comparison intervals from the 200 parameter-bootstrap draws, then assembles English case records. The GWP example uses the route-evaluation table verified by `lca.py`.

Numerical calculations and case assembly run offline. Optional Qwen2-7B generation groups the supplied scientific statements into paragraphs. To use a local checkpoint, install `transformers` and `accelerate`, then run:

```bash
export QWEN_MODEL_PATH=/path/to/local/Qwen2-7B
python analysis/cases.py --regroup
```

Use `--skip-bootstrap` for a faster check of point estimates and case assembly.

## Reactor-model overview

The COMSOL model represents an axisymmetric tubular reactor coupling laminar flow, heat transfer, species transport and reaction source terms. The corresponding graph record describes this model qualitatively.

Outputs are written under `outputs/analysis`.
