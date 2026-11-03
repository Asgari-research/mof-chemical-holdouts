"""Redraw two defective V6 control plots from frozen aggregate CSVs; no fitting.

Run from any directory: python plotting/replot_v6_controls.py
Requires matplotlib, numpy, pandas. Output is written to generated_previews/; locked figures are never overwritten.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_DIR=ROOT/"generated_previews"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#202830',
 'text.color':'#202830','xtick.color':'#202830','ytick.color':'#202830',
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#35638F'; CORAL='#B65E51'
df=pd.read_csv(ROOT/'v6_source_data/matched_random_vs_topology_fold_metrics.csv').sort_values('fold')
tf=pd.read_csv(ROOT/'plotting/data/topology_fold_heterogeneity.csv').sort_values('fold_id')
np.testing.assert_allclose(df.topology_r2,tf.r2,atol=1e-14)
np.testing.assert_allclose(df.topology_top5_recovery,tf.top_5pct_overlap,atol=1e-14)
assert len(df)==len(tf)==5

def format_axes(ax, ylabel, ylim):
 ax.set(xlabel='Outer fold',ylabel=ylabel,ylim=ylim,xticks=np.arange(5),xticklabels=np.arange(5))
 ax.grid(axis='y',color='#E2E7EC',linewidth=.6,zorder=0)
 ax.axhline(0,color='#677583',linewidth=.8)

def annotate(ax,bars,percent=False):
 for b in bars:
  v=b.get_height();ax.annotate(f'{v:.1f}' if percent else f'{v:.3f}',
   (b.get_x()+b.get_width()/2,v),xytext=(0,4 if v>=0 else -5),
   textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=8.2)

def save(fig,stem):
 for ext in ['pdf','svg','png']:
  fig.savefig(OUTPUT_DIR/f'{stem}.{ext}',bbox_inches='tight',dpi=220,transparent=False)
 plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10.6,4.0))
fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.80,wspace=.28)
x=np.arange(5);w=.35
for ax,yr,yt,ylabel,ylim,percent in [
 (axs[0],df.random_r2,df.topology_r2,r'$R^2$',(-.13,.88),False),
 (axs[1],100*df.random_top5_recovery,100*df.topology_top5_recovery,'Fold-local top-5% recovery (%)',(0,73),True)]:
 format_axes(ax,ylabel,ylim)
 b1=ax.bar(x-w/2,yr,w,color=BLUE,label='Size-matched random',zorder=3)
 b2=ax.bar(x+w/2,yt,w,color=CORAL,label='Topology holdout',zorder=3)
 annotate(ax,b1,percent);annotate(ax,b2,percent)
axs[0].set_title('(a) Predictive fit',loc='left',fontsize=12,pad=11)
axs[1].set_title('(b) Candidate recovery',loc='left',fontsize=12,pad=11)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.0))
save(fig,'v6_matched_random_vs_topology')

fig,axs=plt.subplots(1,2,figsize=(10.6,4.0))
fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.85,wspace=.28)
for ax,ys,ylabel,ylim,percent in [
 (axs[0],tf.r2,r'$R^2$',(-.12,.70),False),
 (axs[1],100*tf.top_5pct_overlap,'Fold-local top-5% recovery (%)',(0,70),True)]:
 format_axes(ax,ylabel,ylim)
 bars=ax.bar(x,ys,.55,color=CORAL,zorder=3);annotate(ax,bars,percent)
axs[0].set_title('(a) Topology-held-out predictive fit',loc='left',fontsize=12,pad=11)
axs[1].set_title('(b) Topology-held-out candidate recovery',loc='left',fontsize=12,pad=11)
save(fig,'v6_topology_fold_heterogeneity')
print('Redrawn from five stored fold scores each; negative fold-1 R² retained.')
