"""Explanation and utility audit under simulated quantization.

Fits the three MLPs, simulates each precision and range configuration over
calibration cohorts, and records field-ablation explanations and utility.
Use --pilot for a quick check.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from calibration_explanations import load_data, FEATURES, forward_layers

DEAD_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
CATEGORICAL = ['admission_type_id', 'discharge_disposition_id',
               'A1Cresult', 'change', 'diabetesMed', 'diag_group']
POLICIES = ('representative', 'balanced', 'majority_only')
# Precision and range-rule configurations.
CONFIGS = [
    ('W8A8_pooled_p995', 8, 8, False, 99.5),
    ('W8A8_pooled_minmax', 8, 8, False, 100),
    ('W4A8_pooled_p995', 4, 8, False, 99.5),
    ('W8A4_pooled_p995', 8, 4, False, 99.5),
    ('W4A4_pooled_p995', 4, 4, False, 99.5),
    ('W8A4_channel_p995', 8, 4, True, 99.5),
    ('W8A4_channel_minmax', 8, 4, True, 100),
]


def prepare(cache: Path):
    # Downloads the public UCI archive on first use.
    load_data(cache)
    with zipfile.ZipFile(cache / 'diabetes_130_us_hospitals.zip') as z:
        raw = pd.read_csv(z.open('diabetic_data.csv'), low_memory=False)
    raw = raw.drop_duplicates('patient_nbr', keep='first').reset_index(drop=True)
    raw = raw.loc[~raw.discharge_disposition_id.isin(DEAD_OR_HOSPICE)].reset_index(drop=True)
    raw['age'] = pd.to_numeric(raw.age.str.extract(r'\[(\d+)-')[0]) + 5
    code = pd.to_numeric(raw.diag_1.str.slice(0, 3), errors='coerce')
    raw['diag_group'] = np.where(code.isna(), raw.diag_1.astype(str).str.slice(0, 1),
                                 (code.fillna(0).astype(int) // 50).astype(str))
    labels = (raw.readmitted == '<30').to_numpy(dtype=np.int8)
    return raw, labels


class Quantized:
    """Symmetric per-tensor weights, per-tensor hidden and optional channel input."""
    def __init__(self, model, calibration, wbits, abits, input_channel, percentile):
        self.qw, self.qa = 2 ** (wbits - 1) - 1, 2 ** (abits - 1) - 1
        self.biases = model.intercepts_
        self.weights = []
        for w in model.coefs_:
            scale = max(np.abs(w).max() / self.qw, 1e-12)
            self.weights.append(np.clip(np.rint(w / scale), -self.qw, self.qw) * scale)
        layers = forward_layers(model, calibration)
        input_axis = 0 if input_channel else None
        self.ranges = [np.maximum(np.percentile(np.abs(h), percentile,
                                                 axis=input_axis if i == 0 else None), 1e-9)
                       for i, h in enumerate(layers)]

    def predict(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            scale = self.ranges[i] / self.qa
            x = np.clip(np.rint(x / scale), -self.qa, self.qa) * scale
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)
        return expit(x.ravel())


def grouped_attributions(predict, x, ref, groups):
    base=predict(x)
    out=np.empty((len(x),len(groups)))
    for j, cols in enumerate(groups):
        altered=x.copy()
        altered[:,cols]=ref[cols]
        out[:,j]=base-predict(altered)
    return out


def explain_stats(fp_attr, q_attr, q_predict, x, ref, random_top, groups):
    n = len(x)
    rows = np.arange(n)[:, None]
    # Stable sort removes incidental ordering of exactly tied attribution values.
    sort_fp = np.argsort(-np.abs(fp_attr), axis=1, kind='stable')
    sort_q = np.argsort(-np.abs(q_attr), axis=1, kind='stable')
    eligible = (np.count_nonzero(fp_attr, axis=1) >= 3) & (np.count_nonzero(q_attr, axis=1) >= 3)
    fp_top = sort_fp[:, :3]
    q_top = sort_q[:, :3]
    overlap = np.mean([len(set(a) & set(b)) / 3 for a, b in zip(fp_top[eligible], q_top[eligible])]) if eligible.any() else np.nan
    xa, xr = x.copy(), x.copy()
    for j, cols in enumerate(groups):
        take_top=(q_top == j).any(axis=1)
        take_rand=(random_top == j).any(axis=1)
        xa[np.ix_(take_top, cols)] = ref[cols]
        xr[np.ix_(take_rand, cols)] = ref[cols]
    base = q_predict(x)
    return dict(top3_overlap_eligible=float(overlap),
                eligible_fraction=float(eligible.mean()),
                attr_l1_absolute_mean=float(np.mean(np.abs(fp_attr - q_attr).sum(axis=1))),
                fp_attr_l1_absolute_mean=float(np.mean(np.abs(fp_attr).sum(axis=1))),
                top3_deletion=float(np.abs(base - q_predict(xa)).mean()),
                random3_deletion=float(np.abs(base - q_predict(xr)).mean()))


def choose(cal_pool, y, policy, n, rng):
    if policy == 'representative':
        return rng.choice(cal_pool, n, replace=False)
    pos, neg = cal_pool[y[cal_pool] == 1], cal_pool[y[cal_pool] == 0]
    if policy == 'balanced':
        return np.r_[rng.choice(pos, n // 2, replace=False),
                     rng.choice(neg, n - n // 2, replace=False)]
    return rng.choice(neg, n, replace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('repaired_cohort_results'))
    ap.add_argument('--max-iter', type=int, default=90,
                    help='Iteration cap; 250 is a convergence sensitivity check')
    ap.add_argument('--only-config',choices=[z[0] for z in CONFIGS],
                    help='Run one setting without recomputing the complete ablation')
    args = ap.parse_args()
    args.output.mkdir(exist_ok=True)
    raw, y = prepare(args.output / 'cache')
    # One patient per row; the same split is used for every seed.
    inds = np.arange(len(y))
    tr, remaining = train_test_split(inds, train_size=35000, random_state=824,
                                     stratify=y)
    te, remaining = train_test_split(remaining, train_size=10000,
                                     random_state=825, stratify=y[remaining])
    va, cal = train_test_split(remaining, train_size=3000,
                              random_state=826, stratify=y[remaining])
    assert len(np.unique(np.r_[tr, te, va, cal])) == len(y)
    num = raw[FEATURES].astype(float).to_numpy()
    cat = raw[CATEGORICAL].fillna('Missing').astype(str)
    sc = StandardScaler().fit(num[tr])
    oh = OneHotEncoder(handle_unknown='ignore', min_frequency=100,
                       sparse_output=False).fit(cat.iloc[tr])
    x = np.column_stack((sc.transform(num), oh.transform(cat))).astype(np.float64)
    # Categorical references are whole, valid categories, not per-column medians.
    typical=cat.iloc[tr].mode(dropna=False).iloc[0]
    ref=np.r_[np.median(x[tr,:len(FEATURES)],axis=0),
              oh.transform(pd.DataFrame([typical],columns=CATEGORICAL))[0]]
    names=oh.get_feature_names_out(CATEGORICAL)
    groups=[np.asarray([i]) for i in range(len(FEATURES))]
    for c in CATEGORICAL:
        cols=np.flatnonzero(np.array([z.startswith(c+'_') for z in names]))+len(FEATURES)
        assert len(cols)>0
        groups.append(cols)
    assert sorted(np.concatenate(groups).tolist()) == list(range(x.shape[1]))
    gen = np.random.default_rng(761)
    # Fixed explanation panel and random-field baseline.
    expl = np.r_[gen.choice(te[y[te] == 1], 128, replace=False),
                 gen.choice(te[y[te] == 0], 128, replace=False)]
    rand_top = np.stack([gen.choice(len(groups), 3, replace=False) for _ in expl])
    records, baselines = [], []
    seeds = [311] if args.pilot else [311, 312, 313]
    draws = 2 if args.pilot else 24
    print('patients',len(y),'features',x.shape[1],
          'train/test/validation/calibration',len(tr),len(te),len(va),len(cal),flush=True)
    for seed in seeds:
        model = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=.1,
                              max_iter=args.max_iter, n_iter_no_change=12,
                              random_state=seed)
        model.fit(x[tr], y[tr])
        fp = lambda z: model.predict_proba(z)[:, 1]
        pf = fp(x[te])
        attrf = grouped_attributions(fp,x[expl],ref,groups)
        fp_auroc = roc_auc_score(y[te],pf)
        threshold = float(np.quantile(fp(x[va])[y[va] == 1], .30))
        base_brier = float(y[te].mean() * (1-y[te].mean()))
        baselines.append(dict(seed=seed, fp32_auroc=fp_auroc,
                              fp32_auprc=average_precision_score(y[te],pf),
                              fp32_brier=brier_score_loss(y[te],pf),
                              constant_prevalence_brier=base_brier,
                              positive_rate=float(y[te].mean()),
                              fp32_top3_deletion=float('nan'),
                              fixed_threshold=threshold,
                              iterations=model.n_iter_, final_train_loss=model.loss_))
        print('seed',seed,'FP32 AUROC',round(fp_auroc,4),'Brier',
              round(brier_score_loss(y[te],pf),4),'null Brier',round(base_brier,4),flush=True)
        for draw in range(draws):
            for policy in POLICIES:
                rng = np.random.default_rng(seed * 100000 + draw * 17 + POLICIES.index(policy))
                cohort = choose(cal,y,policy,256,rng)
                for name,wb,ab,channel,percentile in CONFIGS:
                    if args.only_config and name!=args.only_config:
                        continue
                    q = Quantized(model,x[cohort],wb,ab,channel,percentile)
                    pred = q.predict(x[te])
                    aq = grouped_attributions(q.predict,x[expl],ref,groups)
                    metrics = explain_stats(attrf,aq,q.predict,x[expl],ref,rand_top,groups)
                    records.append(dict(seed=seed,draw=draw,policy=policy,
                        config=name,weight_bits=wb,activation_bits=ab,
                        per_feature_input=channel,range_percentile=percentile,
                        calibration_positives=int(y[cohort].sum()),
                        auroc=roc_auc_score(y[te],pred),
                        auprc=average_precision_score(y[te],pred),
                        brier=brier_score_loss(y[te],pred),
                        sensitivity=float(np.mean(pred[y[te] == 1] >= threshold)),
                        threshold_disagreement=float(np.mean((pred >= threshold) != (pf >= threshold))),
                        **metrics))
            if draw % 6 == 0:
                print('seed',seed,'draws complete',draw + 1,'/',draws,flush=True)
    df=pd.DataFrame(records)
    df.to_csv(args.output / ('pilot.csv' if args.pilot else 'all_draws.csv'),index=False)
    pd.DataFrame(baselines).to_csv(args.output / ('pilot_baselines.csv' if args.pilot else 'baselines.csv'),index=False)
    summary=df.groupby(['config','policy'],as_index=False).agg(
        n=('auroc','size'),auroc=('auroc','mean'),brier=('brier','mean'),
        sensitivity=('sensitivity','mean'),threshold_disagreement=('threshold_disagreement','mean'),
        top3_overlap=('top3_overlap_eligible','mean'),eligible_fraction=('eligible_fraction','mean'),
        absolute_attr_drift=('attr_l1_absolute_mean','mean'),top3_deletion=('top3_deletion','mean'))
    summary.to_csv(args.output / ('pilot_summary.csv' if args.pilot else 'summary.csv'),index=False)
    print(summary.to_string(index=False,float_format=lambda z:f'{z:.4f}'),flush=True)


if __name__=='__main__':
    main()
