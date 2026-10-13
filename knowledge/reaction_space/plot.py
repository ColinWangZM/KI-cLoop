from pathlib import Path
import os,json,argparse,hashlib
ROOT=Path(__file__).resolve().parent
(ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/review").mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/review/mplcache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/review/cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from PIL import Image
BLUE='#254D76'
LABELS={'Heteroatom Alkylation and Arylation': 'Heteroatom alkylation\n/ arylation', 'Acylation': 'Acylation', 'C-C Coupling': 'C–C bond formation', 'Aromatic Heterocycle Formation': 'Aromatic heterocycle\nformation', 'Protection': 'Protection', 'Deprotection': 'Deprotection', 'Reduction': 'Reduction', 'Oxidation': 'Oxidation', 'Functional Group Interconversion': 'Functional-group\ninterconversion', 'Functional Group Addition': 'Functional-group\naddition', 'Miscellaneous': 'Unclassified / other'}
plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none',
                     'pdf.fonttype':42,'ps.fonttype':42,'text.color':'#233B4D',
                     'savefig.facecolor':'white','axes.linewidth':.6})
DPI=450

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layout',default='coordinates')
    a=ap.parse_args()
    df=pd.read_csv(ROOT/'data/records.csv')
    classes=pd.read_csv(ROOT/'data/classes.csv')
    z=np.load(ROOT/'data'/f'{a.layout}.npy')
    scores=pd.read_csv(ROOT/'data/support.csv')

    examples=pd.read_csv(ROOT/'data/examples.csv').copy()
    assert list(df.record_id)==list(scores.record_id)
    assert len(df)==len(z) and np.isfinite(z).all()
    ix=examples.point_index.to_numpy()
    assert list(df.record_id.iloc[ix])==list(examples.record_id)
    assert scores.robust_low_support.iloc[ix].all()
    examples['x'],examples['y']=z[ix].T
    examples['annotation_radius']=.023*min(np.ptp(z,axis=0))
    from scipy.spatial.distance import cdist
    cd=cdist(z[ix],z[ix]);np.fill_diagonal(cd,np.inf)
    examples['annotation_radius']=np.minimum(examples.annotation_radius,.38*cd.min(axis=1))
    examples=examples.drop(columns=['projection_radius_k15','projection_radius_percentile'],errors='ignore')
    examples['projection_layout']=a.layout
    examples.to_csv(ROOT/'outputs/compact_low_support_examples.csv',index=False)
    coords=df[['record_id','reaction_class','product_has_organic_peroxide']].copy()
    coords['x'],coords['y']=z.T;coords['robust_low_support']=scores.robust_low_support
    coords.to_csv(ROOT/'outputs/compact_reaction_coordinates.csv',index=False)
    colors=np.array([dict(zip(classes.reaction_class,classes.color))[c] for c in df.reaction_class])
    order=np.random.default_rng(42).permutation(len(df))

    handles=[Line2D([],[],marker='o',linestyle='',markersize=5.8,
                   markerfacecolor=r.color,markeredgecolor='none',label=LABELS[r.reaction_class])
             for r in classes.itertuples()]
    assert handles[-1].get_label()=='Unclassified / other'
    
    lo=z.min(axis=0);hi=z.max(axis=0);margin=.055*(hi-lo)
    xlim=(lo[0]-margin[0],hi[0]+margin[0]);ylim=(lo[1]-margin[1],hi[1]+margin[1])
    def base(ax):
        ax.scatter(*z[order].T,s=4.2,c=colors[order],alpha=.78,linewidths=0)
        ax.set(xlim=xlim,ylim=ylim);ax.set_aspect('equal',adjustable='box');ax.set_axis_off()
    def annotate(ax):
        artists=[]
        for r in examples.itertuples():
            for color,lw,zo in [('white',2.7,6),(BLUE,1.2,7)]:
                c=Circle((r.x,r.y),r.annotation_radius,fill=False,edgecolor=color,linewidth=lw,zorder=zo)
                ax.add_patch(c);artists.append(c)
            t=ax.annotate(r.example,(r.x+r.annotation_radius*.72,r.y+r.annotation_radius*.78),
                          xytext=(2,2),textcoords='offset points',fontsize=8.2,color=BLUE,
                          weight='bold',zorder=8,bbox={'facecolor':'white','edgecolor':'none','pad':.3,'alpha':.85})
            artists.append(t)
        return artists
    def save(fig,kind,variant):
        stem=f'reaction_space_compact_{kind}{variant}'
        for ext in ('png','pdf','svg'):
            fig.savefig(ROOT/'outputs/figures'/f'{stem}.{ext}',dpi=DPI,transparent=ext=='png')
        return stem
    report={'layout':a.layout,'n_records':len(df),'source_checked_example_ids':examples.record_id.tolist(),
            'same_coordinates_for_both_variants':True,'class_offsets_or_manual_coordinate_edits':False,
            'source_coordinates_sha256':hashlib.sha256((ROOT/'data'/f'{a.layout}.npy').read_bytes()).hexdigest(),
            'support_scores_sha256':hashlib.sha256((ROOT/'data/support.csv').read_bytes()).hexdigest(),
            'display_style':'reaction_classes_only',
            'peroxide_specific_markers':False,
            'peroxide_product_records_retained':int(df.product_has_organic_peroxide.sum()),
            'legend_labels':[h.get_label() for h in handles],
            'pairs':{}}
    for kind in ('','map_'):
        if kind=='':
            fig=plt.figure(figsize=(9.05,5.8),dpi=DPI)
            ax=fig.add_axes([.008,.012,.617,.976])
            lax=fig.add_axes([.645,.01,.35,.98]);lax.set_axis_off()
            lax.legend(handles=handles,loc='center left',frameon=False,fontsize=10.6,
                       handletextpad=.6,labelspacing=.8,borderaxespad=0,handlelength=1)
        else:
            fig=plt.figure(figsize=(5.5,5.8),dpi=DPI);ax=fig.add_axes([.012,.012,.976,.976])
        base(ax)
        clean=save(fig,kind,'clean')
        artists=annotate(ax)
        annotated=save(fig,kind,'annotated')
        
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        clean_pixels=np.array(Image.open(ROOT/'outputs/figures'/f'{clean}.png'))
        annotated_pixels=np.array(Image.open(ROOT/'outputs/figures'/f'{annotated}.png'))
        assert clean_pixels.shape==annotated_pixels.shape
        h,w=clean_pixels.shape[:2];mask=np.zeros((h,w),dtype=bool)
        boxes=[]
        for artist in artists:
            bbox=artist.get_window_extent(renderer)
            padding=14 
            x0=max(0,int(np.floor(bbox.x0))-padding);x1=min(w,int(np.ceil(bbox.x1))+padding)
            y0=max(0,h-int(np.ceil(bbox.y1))-padding);y1=min(h,h-int(np.floor(bbox.y0))+padding)
            mask[y0:y1,x0:x1]=True;boxes.append([x0,y0,x1,y1])
        changed=np.any(clean_pixels!=annotated_pixels,axis=2)
        outside=int(np.sum(changed & ~mask));assert outside==0,outside
        assert changed.any()
        report['pairs'][kind or 'combined']={'clean':clean,'annotated':annotated,'pixel_dimensions':[w,h],
                    'changed_pixels_outside_annotations':outside,'changed_pixel_fraction':float(changed.mean()),
                    'annotation_pixel_bounds':boxes,
                    'clean_sha256':hashlib.sha256((ROOT/'outputs/figures'/f'{clean}.png').read_bytes()).hexdigest(),
                    'annotated_sha256':hashlib.sha256((ROOT/'outputs/figures'/f'{annotated}.png').read_bytes()).hexdigest()}
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.35,5.8),dpi=DPI);ax.set_axis_off()
    ax.legend(handles=handles,loc='center left',frameon=False,fontsize=12.5,labelspacing=.75,
              handletextpad=.62,borderaxespad=0,handlelength=1)
    fig.subplots_adjust(left=.015,right=.995,bottom=.01,top=.99)
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/'outputs/figures'/f'reaction_space_compact_legend.{ext}',dpi=DPI,transparent=ext=='png')
    plt.close(fig)
    (ROOT/'outputs/review/compact_pair_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
