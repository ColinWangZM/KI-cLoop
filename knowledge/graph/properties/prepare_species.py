from pathlib import Path
import os
from hashlib import sha256
import json
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import pyarrow.parquet as pq

ROOT=Path(os.environ.get("KICLOOP_PROPERTY_WORK", Path(__file__).resolve().parents[1] / "property_work")).resolve()
BASE=Path(os.environ.get("KICLOOP_GRAPH_WORK", Path(__file__).resolve().parents[1] / "work")).resolve()
SPECIES={
    'H2O2':('Hydrogen peroxide','OO'),
    'TBCL':('tert-Butyl chloride','CC(C)(C)Cl'),
    'TBA':('tert-Butanol','CC(C)(C)O'),
    'TBHP':('tert-Butyl hydroperoxide','CC(C)(C)OO'),
    'DTBP':('Di-tert-butyl peroxide','CC(C)(C)OOC(C)(C)C'),
    'Benzaldehyde':('Benzaldehyde','O=Cc1ccccc1'),
    'TBPB':('tert-Butyl peroxybenzoate','CC(C)(C)OOC(=O)c1ccccc1'),
    'CF3CO2H':('Trifluoroacetic acid','O=C(O)C(F)(F)F'),
    'TBAI':('Tetrabutylammonium iodide','CCCC[N+](CCCC)(CCCC)CCCC.[I-]'),
    'water':('Water','O'),
    'MeCN':('Acetonitrile','CC#N'),
}
records={}
for key,(label,smiles) in SPECIES.items():
    mol=Chem.MolFromSmiles(smiles)
    assert mol is not None
    inchi=Chem.MolToInchi(mol)
    canonical=Chem.MolFromInchi(inchi)
    inchi=Chem.MolToInchi(canonical)
    node_id='molecule:'+sha256(inchi.encode()).hexdigest()
    records[key]={'label':label,'smiles':Chem.MolToSmiles(canonical),'canonical_inchi':inchi,
                  'node_id':node_id,'molar_mass_g_mol':Descriptors.MolWt(canonical),
                  'molar_mass_method':'RDKit MolWt on the canonical structure; calculated, not measured',
                  'rdkit_version':rdBase.rdkitVersion}
    records[key]['formula']=rdMolDescriptors.CalcMolFormula(canonical)
    records[key]['elements']=sorted({a.GetSymbol() for a in canonical.GetAtoms()})
found=set()
for batch in pq.ParquetFile(BASE/'graph_parquet/molecules.parquet').iter_batches(columns=['node_id'],batch_size=100000):
    found.update(set(batch.column(0).to_pylist()) & {row['node_id'] for row in records.values()})
for row in records.values():row['in_existing_graph']=row['node_id'] in found
(ROOT/'inputs/project_species.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(f'{len(records)} canonical project species; {len(found)} linked to molecule records; {len(records)-len(found)} represented by declared structure records.')
