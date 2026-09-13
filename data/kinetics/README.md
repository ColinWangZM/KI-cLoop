# Processed ODE-inverse PINN kinetics data

This directory contains a lightweight analysis snapshot for the four organic
peroxide routes used in the KI-cLoop study:

| Identifier | Route |
|---|---|
| `tbhp_bzcl` | TBHP formation from tert-butyl chloride and hydrogen peroxide |
| `tbhp_wpo4` | Phosphotungstic-acid-catalysed TBHP formation from tert-butanol and hydrogen peroxide |
| `tbhp_cf3` | Trifluoroacetic-acid-catalysed TBHP formation from tert-butanol and hydrogen peroxide |
| `tbpb_benzaldehyde` | TBPB formation from benzaldehyde and TBHP |

## Route-level tables

- `route_kinetic_parameter_summary.csv`: fitted Arrhenius terms, apparent
  reaction orders, product R², and 95% intervals derived from the stored
  bootstrap refits.
- `route_evaluation_summary.csv`: model-recommended route performance and
  normalized evaluation scores.
- `route_condition_comparison.csv`: best measured versus best model condition.
- `route_experimental_design_summary.csv`: number of experiments/time points
  and the sampled temperature, ratio, and catalyst ranges.
- `route_prediction_error_summary.csv`: product MAE, MSE, and RMSE by route.
- `route_product_prediction_pool.csv`: 480 experimental/predicted product
  concentration pairs.
- `condition_optimization_summary.json`: search space, leading candidates,
  measured support, and consensus recommendation for each route.

## Per-route files

For each route, `routes/` contains:

- `*_ode_summary.json`: learned parameters, fit metrics, initialization, and
  training settings;
- `*_ode_predictions.csv`: measured and predicted concentration profiles;
- `*_ode_training_history.csv`: optimization trace;
- `*_bootstrap_params.csv`: residual-bootstrap parameter refits;
- `*_uncertainty_summary.json`: bootstrap parameter/metric summaries;
- `*_profile_likelihood.csv`: one-parameter profile scans.

## Units and interpretation

- temperature fields are in degrees Celsius unless a source field explicitly
  states otherwise;
- time is in seconds in the consolidated optimization/evaluation tables;
- concentrations are in mol/L;
- `Ea*_over_R` is in kelvin and `Ea*_kJ_mol` is in kJ/mol;
- `lnA*` and fitted reaction/catalyst orders are apparent route-specific model
  parameters;
- R² and error metrics describe agreement inside the measured study domain and
  should not be treated as evidence of extrapolation validity.

The uncertainty snapshot uses five residual-bootstrap refits with 20 epochs
per refit. Its intervals are useful for exploratory analysis but should not be
interpreted as high-precision population confidence intervals.

## Quick analysis

```python
import pandas as pd

params = pd.read_csv("data/kinetics/route_kinetic_parameter_summary.csv")
scores = pd.read_csv("data/kinetics/route_evaluation_summary.csv")
pred = pd.read_csv("data/kinetics/route_product_prediction_pool.csv")

print(params[["route_title", "Ea1_kJ_mol", "product_R2"]])
print(scores[["route_title", "overall_score_0_5"]])
print(pred.groupby("route")[["experimental", "predicted"]].corr())
```

Raw experimental workbooks, full training-input tables, and exhaustive
condition-search pools are intentionally not included.
