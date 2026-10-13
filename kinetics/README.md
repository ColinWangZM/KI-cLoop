# Peroxide-route kinetic inference

This module implements the hybrid PINN–ODE model used for four peroxide routes. The input comprises 60 conditions and 420 concentration records: 60 calculated inlet states and 360 noninitial observations. Inlet states supply the initial conditions and are excluded from fitting errors and reported prediction metrics.

| Route | Identifier | Conditions | States | Main apparent activation energy (kJ/mol) | Product R², excluding inlet |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `tbhp_bzcl` | 13 | 91 | 9.0727 | 0.986274 |
| 2 | `tbhp_wpo4` | 15 | 105 | 101.1422 | 0.988612 |
| 3 | `tbhp_cf3` | 17 | 119 | 123.4347 | 0.991984 |
| 4 | `tbpb_benzaldehyde` | 15 | 105 | 4.4949 | 0.927909 |

## Data

`data/<route>_measurements.csv` contains the processed concentration states used for fitting. Each record includes `condition`, `experiment_id`, `time`, `temperature_C`, `catalyst` and the fitted species. Concentrations are in mol/L, time is in seconds and temperature is in degrees Celsius. `experiment_id` groups records belonging to the same experimental condition. The `time = 0` record supplies the inlet concentrations; feed molar ratios can be calculated from these species and catalyst concentrations. Route 1 uses a unit catalyst placeholder because its rate law has no catalyst dependence.

Computational reproduction begins with these concentration states. Species reconstructed through material balances are not independent measurements. Sample preparation and HPLC quantification are described in Supplementary Information section S3.5.

Route 4 uses a TBPB molecular mass of 194.2271 g/mol and an acetonitrile-based TBAI stock. Its fitted model has five parameters and one main reaction.

## Method

An eight-start ODE fit provides trajectories for network pretraining. Joint PINN training then combines concentration data with the rate equations while enforcing initial conditions, nonnegative concentrations and stoichiometric conservation. Finally, ODE refinement minimizes the unweighted sum of squared concentration errors (SSE) over noninitial states. Reported parameters and fit metrics correspond to this final ODE refinement. Route settings are in `routes.json`, and training settings are in `pinn_model.py`.

Routes 2 and 3 use two reaction rates for TBHP and DTBP formation; Routes 1 and 4 use one rate. Arrhenius parameters use R = 8.314 J/(mol K), with prefactor units determined by each fitted rate law.

Conditional parameter intervals use 200 whole-condition wild-residual bootstrap refits of the final ODE estimator, with inlet concentrations fixed. Five held-out-condition folds assess ODE prediction. Parameter profiles, bound and initialization checks, and three neural seeds assess sensitivity and repeatability.

## Run

Install the repository's root `requirements.txt` with Python 3.10 on Linux or WSL and provide a C++17 compiler (`g++`, or set `CXX`). Run from the repository root:

```sh
python kinetics/reproduce.py --mode check
python kinetics/plot_results.py
python kinetics/reproduce.py --mode smoke
python kinetics/reproduce.py --mode baseline
python kinetics/reproduce.py --mode full
```

`check` verifies the model constraints and supplied results. `smoke` runs all four ODE–PINN–ODE paths with shortened training. `baseline` performs the eight-start ODE fits. `primary` runs one complete neural seed with bootstrap and diagnostics; `full` adds two neural seeds and initialization checks.

Use `--output PATH` for a separate output directory and `--resume` to resume a matching run. Resuming requires matching input-file hashes and training settings; use a new output directory when these differ. Generated files go to `outputs/` by default; supplied results are in `results/`.

## Results and manuscript mapping

`results/<route>/final_summary.json` contains the final parameters and metrics, and `final_predictions.csv` pairs observations with ODE predictions. `pinn_summary.json` and `pinn_network_predictions.csv` provide neural estimates and predictions. Bootstrap, grouped-validation and sensitivity results are supplied as CSV/JSON files.

These files support Supplementary Information section S6, Tables 22–24, and the four-route screening calculations. Experimental procedures and concentration records are described in sections S3.1–S3.5 and S5. `plot_results.py` regenerates trajectory, parity and activation-energy interval plots.
