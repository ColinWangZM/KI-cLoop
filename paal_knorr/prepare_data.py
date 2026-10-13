import argparse
from pathlib import Path
import runtime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def derive():
    raw = pd.read_csv(ROOT / 'data/chromatography.csv', float_precision='round_trip')
    rows = []
    for _, group in raw.groupby('experiment_id', sort=False):
        group = group.sort_values('state_index')
        for row in group.itertuples(index=False):
            flow = row.HXD_flow_mL_min + row.EA_flow_mL_min
            hxd_initial = row.HXD_feed_M * row.HXD_flow_mL_min / flow
            ea_initial = row.EA_feed_M * row.EA_flow_mL_min / flow
            extent = hxd_initial - row.HXD_M
            rows.append({'experiment_id': row.experiment_id, 'condition': row.condition, 'state_index': row.state_index,
                         'residence_time_s': row.residence_time_s, 'temperature_C': row.temperature_C,
                         'HXD_M': row.HXD_M, 'EA_derived_M': ea_initial - extent,
                         'DMEP_derived_M': extent, 'independent_fit_observation': row.state_index > 0})
    return pd.DataFrame(rows)


def check():
    derived = derive()
    supplied = pd.read_csv(ROOT / 'data/observations.csv')
    if list(derived.columns) != list(supplied.columns):
        raise AssertionError('Observation schema mismatch.')
    for column in derived:
        if derived[column].dtype.kind in 'biuf':
            np.testing.assert_allclose(derived[column], supplied[column], rtol=1e-12, atol=1e-12)
        elif not derived[column].equals(supplied[column]):
            raise AssertionError(column)
    if len(derived) != 210 or derived.independent_fit_observation.sum() != 180:
        raise AssertionError('Expected 30 inlet states and 180 independent observations.')
    return {'conditions': 30, 'records': 210, 'independent_observations': 180,
            'max_concentration_difference_M': float(abs(derived.HXD_M - supplied.HXD_M).max()),
            'max_time_difference_s': float(abs(derived.residence_time_s - supplied.residence_time_s).max())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(check())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        derive().to_csv(args.output, index=False)
