"""Paired contrasts across calibration policies and precision settings.

Intervals resample calibration draws within the three fitted models. They do
not describe uncertainty across hospitals, data splits, or future patients.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

OUT=Path('repaired_cohort_results')
METRICS=['auroc','brier','sensitivity','threshold_disagreement',
         'top3_overlap_eligible','attr_l1_absolute_mean','top3_deletion']


def summarize(paired, label):
    rng=np.random.default_rng(147)
    out=[]
    for metric in METRICS:
        seed_arrays=[z[metric].to_numpy() for _,z in paired.groupby('seed')]
        if not seed_arrays:
            continue
        estimates=[]
        for rep in range(3000):
            estimates.append(np.mean([rng.choice(v,len(v),replace=True).mean()
                                      for v in seed_arrays]))
        per_seed=[float(v.mean()) for v in seed_arrays]
        seed_se=np.std(per_seed,ddof=1)/np.sqrt(len(per_seed))
        t_margin=t.ppf(.975,len(per_seed)-1)*seed_se
        out.append(dict(contrast=label,metric=metric,
                        n_seeds=len(seed_arrays),n_paired_blocks_each=len(seed_arrays[0]),
                        mean_difference=float(np.mean(per_seed)),
                        conditional_draw_ci_low=float(np.percentile(estimates,2.5)),
                        conditional_draw_ci_high=float(np.percentile(estimates,97.5)),
                        seed_t_ci_low=float(np.mean(per_seed)-t_margin),
                        seed_t_ci_high=float(np.mean(per_seed)+t_margin),
                        min_seed_difference=min(per_seed),
                        max_seed_difference=max(per_seed)))
    return out


def main():
    data=pd.read_csv(OUT/'all_draws.csv')
    result=[]
    # Policy contrasts, paired by seed, draw, and configuration.
    for config in data.config.unique():
        base=data.query('config==@config and policy=="representative"').set_index(['seed','draw'])
        for policy in ('balanced','majority_only'):
            other=data.query('config==@config and policy==@policy').set_index(['seed','draw'])
            result += summarize(other[METRICS]-base[METRICS],f'{config}: {policy} - representative')
    # Precision and range-rule contrasts, paired by seed, draw, and policy.
    for left,right in [('W4A8_pooled_p995','W8A8_pooled_p995'),
                       ('W8A4_pooled_p995','W8A8_pooled_p995'),
                       ('W4A4_pooled_p995','W8A4_pooled_p995'),
                       ('W8A4_channel_p995','W8A4_pooled_p995'),
                       ('W8A4_channel_minmax','W8A4_channel_p995')]:
        a=data.query('config==@left').set_index(['seed','draw','policy'])
        b=data.query('config==@right').set_index(['seed','draw','policy'])
        result += summarize(a[METRICS]-b[METRICS],f'{left} - {right}')
    table=pd.DataFrame(result)
    table.to_csv(OUT/'paired_intervals.csv',index=False)
    print(table.query('metric in ["auroc", "sensitivity", "top3_overlap_eligible"]')
          .to_string(index=False,float_format=lambda v:f'{v:.4f}'))


if __name__=='__main__':
    main()
