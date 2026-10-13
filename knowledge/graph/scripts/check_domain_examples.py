from prepare_molecules import init_worker,process_batch
from prepare_materials import process
from rdkit import Chem

init_worker()
examples=[
    ('TBHP','CC(C)(C)OO',{'organic_peroxide','hydroperoxide'},{'alcohol','perester','peroxyacid'}),
    ('DTBP','CC(C)(C)OOC(C)(C)C',{'organic_peroxide'},{'hydroperoxide','perester'}),
    ('TBPB','CC(C)(C)OOC(=O)c1ccccc1',{'organic_peroxide','perester'},{'hydroperoxide','diacyl_peroxide'}),
    ('Benzoyl peroxide','O=C(c1ccccc1)OOC(=O)c1ccccc1',{'organic_peroxide','diacyl_peroxide'},{'perester','hydroperoxide'}),
    ('Peracetic acid','CC(=O)OO',{'organic_peroxide','peroxyacid'},{'hydroperoxide','perester'}),
    ('Hydrogen peroxide','OO',set(),{'organic_peroxide','hydroperoxide'}),
    ('tert-Butyl hypochlorite','CC(C)(C)OCl',set(),{'organic_peroxide','hydroperoxide','organohalogen'}),
    ('tert-Butanol','CC(C)(C)O',{'alcohol'},{'organic_peroxide'}),
    ('Formaldehyde','C=O',{'aldehyde'},{'ketone'}),
]
for name,smiles,required,forbidden in examples:
    result=process_batch([{'inchi':Chem.MolToInchi(Chem.MolFromSmiles(smiles)),
                           'source_smiles':smiles,'source_name':name}])[0]
    got={item.removeprefix('class:molecule:') for item in result['class_ids']}
    assert required<=got and not forbidden&got,(name,got)
    print('[PASS]',name,sorted(got))
rows=process([{'dataset':'check','record_id':'NaCl','value_raw':'NaCl'},
              {'dataset':'check','record_id':'ethanol','value_raw':'C2H6O'}])
assert set(rows[0]['elements'])=={'Na','Cl'}
assert 'class:material:binary' in rows[0]['class_ids']
assert 'class:material:halogen_containing' in rows[0]['class_ids']
assert 'class:material:oxygen_containing' in rows[1]['class_ids']
assert all('oxide' not in key for row in rows for key in row['class_ids'])
print('[PASS] Composition classes distinguish element presence from compound identity')
