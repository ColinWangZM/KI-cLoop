from pathlib import Path
import os
import sys

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
import duckdb

c = duckdb.connect()
c.execute('SET threads=24')
c.execute("SET memory_limit='24GB'")
source = str(ROOT/'raw_parquet/uspto/*.parquet')
parts = []
for role in ['reactant_','product_','solvent_']:
    parts.append(f'''SELECT trim("{role}inchi") inchi, "{role}smiles" smiles, "{role}name" name
        FROM read_parquet('{source}') WHERE starts_with(trim("{role}inchi"),'InChI=')''')
query = ' UNION ALL '.join(parts)
c.execute(f'''COPY (SELECT inchi,min(smiles) source_smiles,min(name) source_name
    FROM ({query}) GROUP BY inchi ORDER BY inchi)
    TO '{ROOT}/raw_parquet/molecule_inputs.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)''')
print('Distinct source InChIs:',c.execute(f"SELECT count(*) FROM read_parquet('{ROOT}/raw_parquet/molecule_inputs.parquet')").fetchone()[0],flush=True)
