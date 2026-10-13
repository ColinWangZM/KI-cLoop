import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def run(*arguments):
    print('Running: '+' '.join(map(str,arguments)),flush=True)
    subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,check=True)


def verify_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.reader(handle,strict=True)
        header=next(reader,None)
        if not header or any(not name.strip() for name in header) or len(header)!=len(set(header)):
            raise ValueError('Missing or duplicate CSV column names: '+str(path))
        for line,row in enumerate(reader,start=2):
            if len(row)!=len(header):
                raise ValueError(f'CSV column count mismatch: {path}:{line}; expected {len(header)}, found {len(row)}')


def main():
    parser=argparse.ArgumentParser(description='Verify distributed data and recompute the principal numerical results offline.')
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--screening',action='store_true')
    args=parser.parse_args()
    manifest=ROOT/'FILE_MANIFEST.json'
    if manifest.exists():
        expected=json.loads(manifest.read_text())
        for row in expected['files']:
            path=ROOT/row['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('Distributed file is missing or modified: '+row['path'])
        print(f"Verified {len(expected['files'])} distributed file hashes.",flush=True)
        csv_paths=[ROOT/row['path'] for row in expected['files'] if row['path'].endswith('.csv')]
        for path in csv_paths:
            verify_csv(path)
        print(f'Verified the structure of {len(csv_paths)} CSV files.',flush=True)
    run('paal_knorr/run.py','--mode','verify')
    run('kinetics/reproduce.py','--mode','check')
    run('analysis/lca.py','--verify')
    run('analysis/cases.py')
    run('knowledge/build_kinetic_graph.py')
    run('knowledge/verify_application_retrieval.py','--graph','knowledge/outputs/kinetic_graph/graph.sqlite')
    run('knowledge/query.py','For Route 2, what is the main rate multiplier from 60 to 80 degrees Celsius at fixed composition?')
    if args.screening:
        run('analysis/screen.py')
    if args.plots:
        run('kinetics/plot_results.py')
        run('paal_knorr/plot_results.py')
        run('analysis/draw_results.py')
    print('Offline verification completed. Full kinetic neural retraining is a separate workflow.',flush=True)


if __name__=='__main__':
    main()
