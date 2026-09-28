"""Figure 1: top-three overlap and threshold disagreement by precision."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P=Path('repaired_cohort_results')
d=pd.read_csv(P/'all_draws.csv')
d=d[d.policy=='representative']
names=['W8A8_pooled_p995','W4A8_pooled_p995','W8A4_pooled_p995','W4A4_pooled_p995']
labels=['W8A8','W4A8','W8A4','W4A4']
colors=['#156575','#3979ab','#ce7c34','#974c69']
fig,axes=plt.subplots(1,2,figsize=(7.2,3.35),layout='constrained')
for ax,metric,title in [(axes[0],'top3_overlap_eligible','Top-three explanation agreement'),
                        (axes[1],'threshold_disagreement','Decisions crossing FP32 threshold')]:
    means=[]
    lo=[]
    hi=[]
    for name in names:
        by_seed=d[d.config==name].groupby('seed')[metric].mean().to_numpy()
        means.append(by_seed.mean());lo.append(by_seed.mean()-by_seed.min())
        hi.append(by_seed.max()-by_seed.mean())
    x=np.arange(len(names))
    ax.bar(x,means,color=colors,width=.66,edgecolor='white',linewidth=.7)
    ax.errorbar(x,means,yerr=np.array([lo,hi]),fmt='none',ecolor='#222222',
                elinewidth=1,capsize=3,label='Range across 3 models')
    ax.set_xticks(x,labels)
    ax.set_title(title,fontsize=10)
    ax.set_ylabel('Fraction of test patients' if metric=='threshold_disagreement' else 'Overlap (0–1)')
    ax.set_ylim(0,1 if metric=='top3_overlap_eligible' else .23)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.suptitle('Quantization simulation on 15 clinical feature groups',fontsize=11)
fig.supxlabel('Bars: mean across three fitted models. Whiskers: min–max of seed means (24 cohorts per seed).',
              fontsize=7.5)
fig.savefig(P/'precision_explanation_figure.pdf',bbox_inches='tight')
fig.savefig(P/'precision_explanation_figure.png',dpi=220,bbox_inches='tight')
print(P/'precision_explanation_figure.pdf')
