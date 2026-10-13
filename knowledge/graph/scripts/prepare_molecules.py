from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import os
import json
import time
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors

ROOT = Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
RULES = [
    ('alcohol', 'Alcohol group', '[CX4][OX2H]'),
    ('phenol', 'Phenol group', '[c][OX2H]'),
    ('aldehyde', 'Aldehyde group', '[$([CX3H1](=O)[#6]),$([CX3H2]=O)]'),
    ('ketone', 'Ketone group', '[#6][CX3](=O)[#6]'),
    ('carboxylic_acid', 'Carboxylic acid group', '[CX3](=O)[OX2H1]'),
    ('carboxylate', 'Carboxylate group', '[CX3](=O)[O-]'),
    ('ester', 'Carboxylic ester group', '[CX3](=O)[OX2][#6]'),
    ('amide', 'Amide group', '[CX3](=O)[NX3]'),
    ('amine', 'Amine group', '[NX3;!$(N[C,S,P]=[O,S,N]);!$(N~[!#6;!#1])]'),
    ('nitrile', 'Nitrile group', '[CX2]#[NX1]'),
    ('ether', 'Ether group', '[#6;!$(C=O)][OX2][#6;!$(C=O)]'),
    ('organohalogen', 'Carbon–halogen bond', '[#6][F,Cl,Br,I]'),
    ('aromatic', 'Aromatic ring atom', '[a]'),
    ('alkene', 'Non-aromatic carbon–carbon double bond', '[CX3]=[CX3]'),
    ('alkyne', 'Carbon–carbon triple bond', '[CX2]#[CX2]'),
    ('thiol', 'Thiol group', '[#6][SX2H]'),
    ('sulfone', 'Sulfone group', '[#6][SX4](=O)(=O)[#6]'),
    ('organic_peroxide', 'Organic peroxide motif', '[#6][OX2][OX2]'),
    ('hydroperoxide', 'Organic hydroperoxide group', '[#6;!$(C=O)][OX2][OX2H]'),
    ('perester', 'Peroxyester group', '[#6][CX3](=O)[OX2][OX2][#6;!$(C=O)]'),
    ('peroxyacid', 'Peroxycarboxylic acid group', '[#6][CX3](=O)[OX2][OX2H]'),
    ('diacyl_peroxide', 'Diacyl peroxide group', '[#6][CX3](=O)[OX2][OX2][CX3](=O)[#6]'),
]
QUERIES = None

def init_worker():
    global QUERIES
    RDLogger.DisableLog('rdApp.*')
    QUERIES = [(key, Chem.MolFromSmarts(smarts)) for key, _, smarts in RULES]
    assert all(query is not None for _, query in QUERIES)

def process_batch(rows):
    result = []
    for row in rows:
        out = {**row, 'node_id': None, 'canonical_inchi': None, 'canonical_smiles': None,
               'formula': None, 'elements': [], 'class_ids': [], 'validation_status': 'invalid_inchi'}
        try:
            mol = Chem.MolFromInchi(row['inchi'], sanitize=True, removeHs=True)
            if mol is not None:
                canonical = Chem.MolToInchi(mol)
                if canonical:
                    out.update(node_id='molecule:' + sha256(canonical.encode()).hexdigest(),
                               canonical_inchi=canonical,
                               canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                               formula=rdMolDescriptors.CalcMolFormula(mol),
                               elements=sorted({atom.GetSymbol() for atom in mol.GetAtoms()}),
                               class_ids=['class:molecule:' + key for key, query in QUERIES
                                          if mol.HasSubstructMatch(query)],
                               validation_status='rdkit_inchi_validated')
        except Exception as exc:
            out['validation_status'] = 'error:' + type(exc).__name__
        result.append(out)
    return result

def main():
    start = time.time()
    infile = ROOT / 'raw_parquet/molecule_inputs.parquet'
    outfile = ROOT / 'raw_parquet/molecule_identity_map.parquet'
    schema = pa.schema([(x, pa.string()) for x in ['inchi', 'source_smiles', 'source_name', 'node_id',
                        'canonical_inchi', 'canonical_smiles', 'formula']]
                       + [('elements', pa.list_(pa.string())), ('class_ids', pa.list_(pa.string())),
                          ('validation_status', pa.string())])
    total = pq.ParquetFile(infile).metadata.num_rows
    batches = (batch.to_pylist() for batch in pq.ParquetFile(infile).iter_batches(batch_size=1000))
    good = 0
    done = 0
    last_log = start
    with ProcessPoolExecutor(max_workers=24, initializer=init_worker) as pool:
        with pq.ParquetWriter(outfile, schema, compression='zstd') as writer:
            for rows in pool.map(process_batch, batches, chunksize=1):
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                done += len(rows)
                good += sum(row['node_id'] is not None for row in rows)
                if time.time() - last_log >= 10 or done == total:
                    print(f'[molecules] {done:,}/{total:,} ({done/total:.1%}); valid={good:,}; '
                          f'elapsed={time.time()-start:.0f}s', flush=True)
                    last_log = time.time()
    report = {'input_identities': total, 'valid_source_identities': good,
              'invalid_source_identities': total-good, 'rdkit_version': rdBase.rdkitVersion,
              'elapsed_seconds': round(time.time()-start, 2), 'workers': 24,
              'identity_rule': 'RDKit-sanitized canonical InChI, SHA-256 identifier; source-to-canonical mapping retained',
              'classification': 'Nonexclusive SMARTS matches; structural motifs, not reaction or mechanistic labels',
              'rules': [{'class_id': 'class:molecule:'+key, 'label': label, 'smarts': smarts}
                        for key, label, smarts in RULES]}
    (ROOT / 'audit/molecule_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    main()
