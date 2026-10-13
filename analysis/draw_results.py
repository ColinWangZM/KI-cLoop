import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ORDER=['tbhp_bzcl','tbhp_wpo4','tbhp_cf3','tbpb_benzaldehyde']

def main():
    p=argparse.ArgumentParser(description='Draw compact numerical summaries from the distributed results.')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/figures')
    args=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#3378a5','#29977c','#d49632','#a96599']
    fig,axes=plt.subplots(2,2,figsize=(8,6.5),constrained_layout=True)
    for i,key in enumerate(ORDER):
        df=pd.read_csv(ROOT/'kinetics/results'/key/'final_predictions.csv')
        df=df[df.time>0]
        product='TBPB' if i==3 else 'TBHP'
        axes[0,0].scatter(df[product],df[product+'_pred'],s=13,alpha=.75,label=f'Route {i+1}',c=colors[i])
    lim=max(axes[0,0].get_xlim()[1],axes[0,0].get_ylim()[1])
    axes[0,0].plot([0,lim],[0,lim],c='0.5',lw=1)
    axes[0,0].set(xlabel='Observed product (mol/L)',ylabel='Predicted product (mol/L)',title='Product concentration')
    axes[0,0].legend(frameon=False,fontsize=7)
    screening=pd.read_csv(HERE/'reference/screened_optimal_conditions.csv')
    x=np.arange(4)
    axes[0,1].bar(x,screening['yield']*100,color=colors)
    axes[0,1].set(xticks=x,xticklabels=['1','2','3','4'],xlabel='Route',ylabel='Predicted yield (%)',title='Finite condition screening')
    lca=pd.read_csv(HERE/'reference/lca_summary.csv')
    axes[1,0].bar(x,lca.Final_GWP,color=colors)
    axes[1,0].set(xticks=x,xticklabels=['1','2','3','4'],xlabel='Route',ylabel='kg CO2-eq / kg target product',title='Screening GWP estimate')
    scores=pd.read_csv(HERE/'reference/route_evaluation_summary.csv')
    axes[1,1].bar(x,scores.overall_score,color=colors)
    axes[1,1].set(xticks=x,xticklabels=['1','2','3','4'],xlabel='Route',ylabel='Weighted score',title='Five-index route evaluation')
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/'numerical_overview.png',dpi=250)
    fig.savefig(args.output/'numerical_overview.pdf')
    plt.close(fig)
    print('Saved numerical_overview.png and numerical_overview.pdf.')

if __name__=='__main__':main()
