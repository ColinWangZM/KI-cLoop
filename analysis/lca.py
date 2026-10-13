import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORDER = ['tbhp_bzcl', 'tbhp_wpo4', 'tbhp_cf3', 'tbpb_benzaldehyde']
MASS = {'H2O2': 34.01, 'TBCL': 92.567, 'TBA': 74.122, 'TBHP': 90.12, 'DTBP': 146.2273, 'Benzaldehyde': 106.12, 'TBPB': 194.2271, 'WPO4': 2988.13, 'CF3COOH': 114.023, 'TBAI': 369.37}
NAME = {'H2O2': 'Hydrogen peroxide (H2O2, pure equivalent)', 'TBCL': 'tert-Butyl chloride (TBCL)', 'TBA': 'tert-Butanol (TBA)', 'TBHP': 'tert-Butyl hydroperoxide (TBHP)', 'Benzaldehyde': 'Benzaldehyde', 'WPO4': 'Phosphotungstic acid hydrate catalyst', 'CF3COOH': 'Trifluoroacetic acid catalyst', 'TBAI': 'Tetrabutylammonium iodide (TBAI)', 'DTBP': 'Di-tert-butyl peroxide (DTBP)'}

def calculate(conditions, factors):
    factor = factors.set_index('item').factor.to_dict()
    summary, terms, recycling = [], [], []
    for i, row in enumerate(conditions.itertuples(index=False)):
        route = f'Route {i+1}'
        denominator = row.product * MASS['TBPB' if i == 3 else 'TBHP']
        if denominator <= 0:
            raise ValueError('Product mass must be positive.')
        reactants = [('H2O2', 'TBCL'), ('H2O2', 'TBA'), ('H2O2', 'TBA'), ('Benzaldehyde', 'TBHP')][i]
        changes, amounts = [], {}
        for species in reactants:
            initial = getattr(row, species+'_0')
            final = getattr(row, species+'_final')
            if not 0 <= final <= initial+1e-10:
                raise ValueError('Invalid nonnegative stoichiometric inventory.')
            changes.append(initial-final)
            initial_mass = initial*MASS[species]/denominator
            remaining = final*MASS[species]/denominator
            amounts[NAME[species]] = initial_mass-.95*remaining
            recycling.append(dict(route=route, species=species, initial_kg=initial_mass, remaining_kg=remaining, recycle_kg=.95*remaining, waste_kg=.05*remaining, fresh_kg=amounts[NAME[species]]))
        side = row.DTBP_final if i in (1, 2) else 0.
        expected = [row.product, row.product] if i == 0 else [row.product+side, row.product+2*side] if i in (1, 2) else [row.product, 2*row.product]
        if max(abs(np.asarray(changes)-expected)) > 1e-7:
            raise ValueError('Stoichiometric inventory does not close.')
        amounts['Reaction waste treatment'] = sum(x['waste_kg'] for x in recycling if x['route'] == route)
        if i:
            catalyst = ['WPO4', 'CF3COOH', 'TBAI'][i-1]
            amounts[NAME[catalyst]] = row.catalyst*MASS[catalyst]/denominator
            if i == 3:
                amounts['Acetonitrile'] = amounts[NAME[catalyst]]*(1/.3888-1)
        bg, positive, negative = 0., 0., 0.
        for item, amount in amounts.items():
            f = factor[item]
            if item in [NAME['H2O2'], NAME['CF3COOH'], 'Benzaldehyde', 'Acetonitrile', 'Reaction waste treatment']:
                bg += amount*f
                component = 'background estimate'
            else:
                if item in [NAME['WPO4'], NAME['TBAI']]:
                    amount *= .05
                positive += amount*f
                component = 'positive supplement'
            terms.append(dict(route=route,item=item,amount=amount,factor=f,contribution=amount*f,component=component))
            if item == NAME['CF3COOH']:
                credit=.95*amount*f
                negative += credit
                terms.append(dict(route=route,item=item+' 95% recycle credit',amount=.95*amount,factor=f,contribution=-credit,component='negative contribution'))
        if i:
            species = 'DTBP' if i in (1,2) else 'TBA'
            amount=(side if i in (1,2) else row.product)*MASS[species]/denominator
            f=factor[NAME[species]]
            negative += amount*f
            terms.append(dict(route=route,item=NAME[species]+' byproduct credit',amount=amount,factor=f,contribution=-amount*f,component='negative contribution'))
        summary.append(dict(Route=route,Background_GWP=bg,Positive_supplement=positive,Byproduct_and_recycle_credit=negative,Net_supplement=positive-negative,Final_GWP=bg+positive-negative,Unit='kg CO2-Eq/kg target product'))
    return pd.DataFrame(summary), pd.DataFrame(terms), pd.DataFrame(recycling)

def evaluate(conditions, summary, results):
    frame=conditions.copy()
    frame['Ea1_kJ_mol']=[json.loads((results/key/'final_summary.json').read_text())['parameters']['Ea1_kJ_mol'] for key in ORDER]
    frame['gwp_kg_CO2eq_kg']=summary.Final_GWP.to_numpy()
    def scale(values,inverse=False):
        span=values.max()-values.min()
        normalized=np.zeros(len(values)) if span<1e-12 else (values-values.min())/span
        return np.full(len(values),5.) if span<1e-12 else 3+2*(1-normalized if inverse else normalized)
    frame['yield_score']=3+2*frame['yield']
    frame['selectivity_score']=3+2*frame.selectivity
    frame['productivity_score']=scale(frame.productivity_mol_l_s)
    frame['low_Ea_score']=scale(frame.Ea1_kJ_mol,True)
    frame['gwp_score']=scale(frame.gwp_kg_CO2eq_kg,True)
    cols=['yield_score','selectivity_score','productivity_score','low_Ea_score','gwp_score']
    frame['overall_score']=sum(w*frame[col] for w,col in zip([.4,.25,.15,.1,.1],cols))
    frame['technical_score']=sum(w*frame[col] for w,col in zip([.4,.25,.15,.1],cols[:4]))/.9
    return frame

def main():
    p=argparse.ArgumentParser(description='Calculate screening life-cycle inventories and route scores.')
    p.add_argument('--conditions',type=Path,default=HERE/'reference/screened_optimal_conditions.csv')
    p.add_argument('--results',type=Path,default=ROOT/'kinetics/results')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/lca')
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    conditions=pd.read_csv(args.conditions)
    assert conditions.route.tolist()==ORDER
    summary,terms,recycle=calculate(conditions,pd.read_csv(HERE/'data/lca_factors.csv'))
    evaluation=evaluate(conditions,summary,args.results)
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/'summary.csv',index=False)
    terms.to_csv(args.output/'contributions.csv',index=False)
    recycle.to_csv(args.output/'recovery.csv',index=False)
    evaluation.to_csv(args.output/'route_evaluation.csv',index=False)
    sensitivity=[]
    route4=terms[terms.route=='Route 4'].set_index('item')
    for b in [.5,1.,1.5]:
        for a in [.5,1.,1.5]:
            altered=summary.copy()
            altered.loc[3,'Final_GWP'] += (b-1)*route4.loc['Benzaldehyde','contribution']+(a-1)*route4.loc['Acetonitrile','contribution']
            scores=evaluate(conditions,altered,args.results)
            for i,key in enumerate(ORDER):
                sensitivity.append(dict(benzaldehyde_factor_scale=b,acetonitrile_factor_scale=a,route=key,gwp=altered.loc[i,'Final_GWP'],composite=scores.loc[i,'overall_score']))
    pd.DataFrame(sensitivity).to_csv(args.output/'factor_sensitivity.csv',index=False)
    if args.verify:
        ref=pd.read_csv(HERE/'reference/lca_summary.csv')
        np.testing.assert_allclose(summary.Final_GWP,ref.Final_GWP,rtol=1e-12,atol=1e-12)
        ref=pd.read_csv(HERE/'reference/route_evaluation_summary.csv')
        np.testing.assert_allclose(evaluation.overall_score,ref.overall_score,rtol=1e-12,atol=1e-12)
    print(summary[['Route','Final_GWP']].to_string(index=False))

if __name__=='__main__':main()
