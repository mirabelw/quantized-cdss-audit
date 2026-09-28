"""Simulated INT8 scale matching for calibration-set membership.

Uses signed symmetric INT8 scales. The hidden-layer test assumes access to the
FP32 parent network.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from calibration_explanations import FEATURES,forward_layers
from repaired_cohort_study import prepare,CATEGORICAL

OUT=Path('repaired_cohort_results')


def count_rows(rows,seed,size,draw,method,mode,hit,rare):
    truth=np.arange(2*size)<size
    for membership,mask1 in [('member',truth),('nonmember',~truth)]:
        for group,mask2 in [('all',np.ones(len(hit),bool)),('rare',rare),('other',~rare)]:
            mask=mask1&mask2
            rows.append(dict(seed=seed,size=size,draw=draw,method=method,access=mode,
                             membership=membership,group=group,
                             flagged=int(hit[mask].sum()),total=int(mask.sum())))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-iter',type=int,default=90)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True)
    raw,y=prepare(OUT/'cache')
    inds=np.arange(len(y))
    tr,rem=train_test_split(inds,train_size=35000,random_state=824,stratify=y)
    te,rem=train_test_split(rem,train_size=10000,random_state=825,stratify=y[rem])
    va,cal=train_test_split(rem,train_size=3000,random_state=826,stratify=y[rem])
    num=raw[FEATURES].astype(float).to_numpy()
    cat=raw[CATEGORICAL].fillna('Missing').astype(str)
    sc=StandardScaler().fit(num[tr])
    oh=OneHotEncoder(handle_unknown='ignore',min_frequency=100,sparse_output=False).fit(cat.iloc[tr])
    x=np.column_stack((sc.transform(num),oh.transform(cat))).astype(np.float64)
    rows=[]; diagnostics=[]
    for seed in (311,312,313):
        m=MLPClassifier(hidden_layer_sizes=(32,16),alpha=.1,max_iter=args.max_iter,
                        n_iter_no_change=12,random_state=seed).fit(x[tr],y[tr])
        train_layers=forward_layers(m,x[tr]);cal_layers=forward_layers(m,x[cal])
        # Rare-activation thresholds come from training patients.
        rare_threshold=np.asarray([np.quantile(np.abs(h).max(axis=1),.95)
                                   for h in train_layers])
        maxima=np.stack([np.abs(h).max(axis=1) for h in cal_layers],axis=1)
        rare=np.any(maxima>=rare_threshold,axis=1)
        rng=np.random.default_rng(30000+seed)
        for size in (64,256):
            for draw in range(150):
                ids=rng.choice(len(cal),2*size,replace=False)
                member=ids[:size]
                for method,pct in [('minmax',100),('p99_5',99.5)]:
                    # Scales are stored as FP32. Both rules use the same
                    # exact-match test; percentile bounds match only via ties.
                    bounds=np.asarray([np.percentile(np.abs(h[member]),pct)
                                       for h in cal_layers])
                    stored=(bounds/127).astype(np.float32).astype(float)*127
                    hit=np.stack([np.isclose(np.abs(h[ids]),b,
                                              rtol=1e-6,atol=1e-7).any(axis=1)
                                  for h,b in zip(cal_layers,stored)],axis=1)
                    count_rows(rows,seed,size,draw,method,'pooled_input_scale',hit[:,0],rare[ids])
                    count_rows(rows,seed,size,draw,method,'all_scales_with_fp32_parent',hit.any(axis=1),rare[ids])
                    for layer,(h,b) in enumerate(zip(cal_layers,stored)):
                        diagnostics.append(dict(seed=seed,size=size,draw=draw,
                            method=method,layer=layer,
                            boundary_has_member_activation=bool(np.isclose(
                                np.abs(h[member]),b,rtol=1e-6,atol=1e-7).any()),
                            boundary_matches_nonmember=bool(np.isclose(
                                np.abs(h[ids[size:]]),b,rtol=1e-6,atol=1e-7).any())))
        print('completed seed',seed,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(args.output/'stronger_privacy_trials_v2.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(args.output/'percentile_boundary_diagnostics.csv',index=False)
    s=df.groupby(['size','method','access','group','membership'],as_index=False)[['flagged','total']].sum()
    s['rate']=s.flagged/s.total
    s.to_csv(args.output/'stronger_privacy_summary_v2.csv',index=False)
    print(s.query('group in ["all", "rare"]').to_string(index=False,
          float_format=lambda z:f'{z:.4f}'))


if __name__=='__main__':
    main()
