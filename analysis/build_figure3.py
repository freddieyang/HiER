"""Plot paired mean positive ranks for every archived test query."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/hier_rank_effect_mpl'
from pathlib import Path
import argparse
from io_utils import prepare_output
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = prepare_output(args.output_dir, [args.data_dir])
    pairs = pd.read_csv(args.data_dir/'figure3_rank_pairs.csv')
    summary = pd.read_csv(args.data_dir/'figure3_rank_summary.csv').set_index('direction')
    COLORS = {'Improved':'#347F69', 'Unchanged':'#A9B1B5', 'Worsened':'#C57550'}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7, 'axes.linewidth':.6,
                         'axes.labelsize':7, 'xtick.labelsize':6.2, 'ytick.labelsize':6.2,
                         'text.color':'#263A43', 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(3.44,2.37))
    audit=[]
    for ax,direction,title in zip(axes,['E-R','R-E'],['a  Enzyme → reaction','b  Reaction → enzyme']):
        data=pairs[pairs.direction==direction]
        row=summary.loc[direction]; n,c=int(row.N),int(row.C)
        assert len(data)==data.query_id.nunique()==n
        assert data.seed_count.eq(3).all()
        assert data.rank_change.isin(COLORS).all()
        assert ((data.latent_mr >= 1) & (data.full_mr >= 1)).all()
        assert np.allclose([row[cat.lower()+"_percent"] for cat in COLORS],
                           [100*data.rank_change.eq(cat).mean() for cat in COLORS])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.plot([1,c],[1,c],ls='--',lw=.65,color='#67777E',zorder=1)
        for category in ['Unchanged','Worsened','Improved']:
            group=data[data.rank_change==category]
            artist=ax.scatter(group.latent_mr,group.full_mr,s=3.2 if direction=='R-E' else 2.2,
                              color=COLORS[category],alpha=.60,linewidths=0,zorder=2,rasterized=False)
            assert len(artist.get_offsets())==len(group)
            audit.append({'direction':direction,'category':category,'plotted_queries':len(group)})
        ax.set_xlim(.85,c*1.08); ax.set_ylim(.85,c*1.08)
        ticks=[1,10,100] if direction=='E-R' else [1,100,10000]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=2,width=.6,pad=2)
        ax.set_box_aspect(1)
        ax.set_xlabel('Latent rank',labelpad=3)
        ax.set_title(title,fontsize=6.8,fontweight='bold',loc='left',pad=21)
        ax.text(0,1.035,f'N={n:,}; C={c:,}',transform=ax.transAxes,fontsize=6.0,va='bottom')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for pos,category in zip([.08,.50,.92],['Improved','Unchanged','Worsened']):
            color=COLORS[category] if category!='Unchanged' else '#6B777E'
            ax.text(pos,-.40,f'{row[category.lower()+"_percent"]:.1f}%',transform=ax.transAxes,
                    ha='center',fontsize=6.6,color=color,fontweight='bold')
    axes[0].set_ylabel('Full rank',labelpad=3)
    handles=[Line2D([],[],marker='o',ls='',color=color,ms=3,label=cat) for cat,color in COLORS.items()]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.015),frameon=False,
               fontsize=6.5,handlelength=.7,handletextpad=.35,columnspacing=1.1)
    fig.subplots_adjust(left=.14,right=.988,bottom=.23,top=.80,wspace=.47)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(args.output_dir/f'figure3_rank_effect.{ext}',dpi=400,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    (args.output_dir/'figure3_display_validation.json').write_text(json.dumps({'all_queries':True,
            'queries_per_panel':summary.N.astype(int).to_dict(),'seeds':[42,43,44],
            'subsampling':False,'jitter':False,'rank_definition':'all-positive query mean, then mean across seeds',
            'scatter_counts':audit},indent=2)+'\n')
    print('Rendered all',len(pairs),'queries; two directions; three paired seeds.')


if __name__ == "__main__":
    main()
