# Paal–Knorr experimental validation

This module reproduces the hybrid PINN–ODE analysis of 2,5-hexanedione (HXD) and ethanolamine (EA) in methanol. It provides the validation case used before applying the framework to the peroxide synthesis routes. The model describes HXD consumption with an apparent power law, `r = A exp(-Ea / RT) [HXD]^a [EA]^b`, with concentrations in mol/L and time in seconds.

## Data

`data/observations.csv` contains 30 experimental conditions at six temperatures (34.5–82.0 °C) and five inlet compositions. The 180 noninitial HXD observations are the independent fitting observations. EA and DMEP concentrations are derived by 1:1 stoichiometry; they are not additional independent measurements. Inlet states are excluded from the fitting loss and noninitial evaluation metrics.

`data/chromatography.csv` contains processed HXD concentrations, feed concentrations, flow rates, temperatures, residence times and quantification-source labels. Flow rates are reported to three decimal places. `prepare_data.py` uses the tabulated `HXD_M` values directly and derives inlet concentrations from the feed concentrations and flow rates. HXD consumption then determines EA and DMEP concentrations by stoichiometry. Column names specify units. Analytical quantification is described in Supplementary Information section S3.6.

The `conversion_source` and `conversion_aggregation` fields specify how each observation was quantified. Most observations use the mean of two internal-standard estimates. P15 states 1–6 use recorded conversion estimates; P30 state 6 uses only the DMP internal standard.

## Run

Use Python 3.10, the repository's kinetic-analysis dependencies, and a C++17 compiler (`g++` by default, or set `CXX`).

From the repository root:

```bash
python paal_knorr/run.py --mode verify
python paal_knorr/prepare_data.py --output paal_knorr/outputs/reconstructed_observations.csv
python paal_knorr/plot_results.py
```

Verification checks all 210 records against the tabulated times and concentrations and verifies ODE predictions. Plotting regenerates the 30 conversion trajectories and the fit and grouped-validation parity panels.

Full estimation:

```bash
python paal_knorr/run.py --mode all
```

This runs eight ODE initializations, 200 bootstrap refits, grouped validation, and three neural seeds. Individual stages use `--mode baseline`, `--mode diagnostics`, and `--mode train`; a short training check uses `--mode smoke`. Diagnostics and training use the generated baseline when available, otherwise the supplied baseline. Use `--output PATH` to select a separate result directory.

## Model and estimator

An eight-start ODE fit provides trajectories for network pretraining. Joint PINN training combines concentration data, ODE residuals and the baseline trajectories while enforcing initial conditions, nonnegative concentrations and stoichiometry. The learned kinetic parameters then undergo final ODE least-squares refinement, which supplies the reported estimates.

Equally weighted HXD and derived EA concentrations give the same least-squares solution as HXD alone, with 180 independent noninitial observations. `reference_results/seed_comparisons.json` reports the network, PINN-parameter ODE and final ODE estimates separately.

Conditional parameter intervals use 200 whole-condition wild-residual bootstrap refits of the final ODE estimator. Grouped validation refits the ODE model while holding out entire temperature groups (six folds) or feed-ratio groups (five folds). Three neural seeds assess repeatability.

## Reference results

| Quantity | Final estimate |
| --- | ---: |
| HXD order, a | 1.140846 |
| EA order, b | 1.269886 |
| Apparent activation energy (kJ/mol) | 12.327103 |
| ln(A), numerical mol/L and s convention | 0.376653 |
| Noninitial HXD concentration R² | 0.992250 |
| Noninitial conversion R² | 0.964360 |
| Noninitial conversion RMSE (percentage points) | 3.158097 |
| Leave-temperature-out conversion R² | 0.951105 |
| Leave-ratio-out conversion R² | 0.963020 |

`reference_results/parameters.csv`, `metrics.csv`, `fold_metrics.csv`, and `seed_parameters.csv` contain parameter intervals, fit and validation metrics, and neural-seed comparisons. `predictions.csv` provides observations and fitted and held-out predictions; `bootstrap_parameters.csv` contains all 200 refits. The conditional activation-energy interval is 10.982–14.180 kJ/mol.

Moore and Jensen reported a first-step activation energy of 12.2 ± 0.4 kJ/mol in “Batch” Kinetics in Flow: Online IR Analysis and Continuous Control, *Angewandte Chemie International Edition* **53**, 470–473 (2014), [doi:10.1002/anie.201306468](https://doi.org/10.1002/anie.201306468). Comparison with their two-step DMSO model assesses activation-energy consistency across different solvents and model definitions; the present model describes apparent HXD consumption in methanol.
