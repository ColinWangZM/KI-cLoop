from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import json
import time
import pyarrow as pa
import pyarrow.parquet as pq
from pymatgen.core import Composition

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
FORMULAS = {'oqmd':'chemical_formula_reduced', 'material_project':'formula_pretty',
            'jarvis_data':'formula', 'qmof':'formula'}

def process(rows):
    for row in rows:
        row['node_id'] = 'material:' + row['dataset'] + ':' + row['record_id']
        row['formula_raw'] = row.pop('value_raw')
        row['elements'] = []
        row['class_ids'] = []
        row['validation_status'] = 'unparsed_formula'
        try:
            composition = Composition(row['formula_raw'], strict=True)
            row['elements'] = sorted(element.symbol for element in composition.elements)
            count = len(row['elements'])
            if count:
                key = 'elemental' if count == 1 else 'binary' if count == 2 else 'ternary' if count == 3 else 'four_or_more_elements'
                row['class_ids'].append('class:material:' + key)
                row['validation_status'] = 'formula_parsed'
                for symbol, name in [('O','oxygen_containing'), ('C','carbon_containing'), ('N','nitrogen_containing')]:
                    if symbol in row['elements']:
                        row['class_ids'].append('class:material:'+name)
                if set(row['elements']) & {'F','Cl','Br','I','At','Ts'}:
                    row['class_ids'].append('class:material:halogen_containing')
        except Exception:
            pass
        
        if row['dataset'] == 'qmof':
            row['class_ids'].append('class:material:qmof_framework_record')
    return rows

def main():
    start = time.time()
    schema = pa.schema([(x, pa.string()) for x in ['dataset','record_id','property_name',
                      'source_member']] + [('source_row',pa.int64())]
                      + [(x,pa.string()) for x in ['node_id','formula_raw']]
                      + [('elements',pa.list_(pa.string())),('class_ids',pa.list_(pa.string())),
                         ('validation_status',pa.string())])
    batches = []
    for dataset, field in FORMULAS.items():
        for batch in pq.ParquetFile(ROOT/f'raw_parquet/materials/{dataset}__{field}.parquet').iter_batches(batch_size=2000):
            batches.append(batch.to_pylist())
    count = invalid = 0
    with ProcessPoolExecutor(max_workers=12) as pool:
        with pq.ParquetWriter(ROOT/'raw_parquet/material_features.parquet',schema,compression='zstd') as writer:
            for rows in pool.map(process,batches):
                writer.write_table(pa.Table.from_pylist(rows,schema=schema))
                count += len(rows)
                invalid += sum(row['validation_status'] != 'formula_parsed' for row in rows)
                if count % 100000 < 2000:
                    print(f'[materials] {count:,} processed; unparsed={invalid}; elapsed={time.time()-start:.0f}s',flush=True)
    (ROOT/'audit/material_classification.json').write_text(json.dumps({
        'records':count,'unparsed_formulas':invalid,'elapsed_seconds':round(time.time()-start,2),
        'classes':'Element-count and element-presence classes; QMOF framework membership from dataset provenance',
        'notes':'Nonexclusive. Oxygen-containing does not imply oxide; carbon-containing does not imply organic. '
                'Records remain dataset-scoped; polymorphs and cross-dataset duplicates are not collapsed by formula.'
    },indent=2)+'\n')
    print(f'[complete] {count:,} material records; {invalid} unparsed formulas',flush=True)

if __name__ == '__main__':
    main()
