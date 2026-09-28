"""Shared data loading and model helpers.

The other scripts import load_data, FEATURES, and forward_layers from here.
This file's own command-line entry point runs an earlier nine-feature pilot
that is not used in the paper.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
FEATURES = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def load_data(cache: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    local_zip = cache / "diabetes_130_us_hospitals.zip"
    if not local_zip.exists():
        print("Downloading public UCI dataset ...", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "diabetic_data.csv" not in archive.namelist():
                raise RuntimeError("Unexpected dataset archive")
        local_zip.write_bytes(data)
    with zipfile.ZipFile(local_zip) as archive:
        df = pd.read_csv(archive.open("diabetic_data.csv"), low_memory=False)
    # Keep the first listed encounter per patient before any split.
    df = df.drop_duplicates("patient_nbr", keep="first").reset_index(drop=True)
    age_lower = pd.to_numeric(df["age"].str.extract(r"\[(\d+)-")[0])
    df["age"] = age_lower + 5
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        raise RuntimeError("Unexpected missing values in the nine chosen features")
    y = (df["readmitted"] == "<30").to_numpy(dtype=np.int8)
    return x, y


def patient_partitions(y: np.ndarray) -> dict[str, np.ndarray]:
    remaining = np.arange(len(y))
    parts = {}
    for name, count in [("test", 10000), ("calibration", 10000),
                        ("validation", 3000), ("train", 10000)]:
        selected, remaining = train_test_split(
            remaining, train_size=count, stratify=y[remaining], random_state=2027
        )
        parts[name] = np.asarray(selected)
    # Remaining patients are unused.
    assert len(set(np.concatenate(list(parts.values())))) == sum(map(len, parts.values()))
    return parts


def cohort(indices: np.ndarray, y: np.ndarray, policy: str, n: int,
           rng: np.random.Generator) -> np.ndarray:
    pos, neg = indices[y[indices] == 1], indices[y[indices] == 0]
    if policy == "representative":
        return rng.choice(indices, n, replace=False)
    if policy == "balanced":
        return np.concatenate((rng.choice(pos, n // 2, replace=False),
                               rng.choice(neg, n - n // 2, replace=False)))
    if policy == "majority_only":
        return rng.choice(neg, n, replace=False)
    raise ValueError(policy)


def forward_layers(model: MLPClassifier, x: np.ndarray) -> tuple[np.ndarray, ...]:
    h1 = np.maximum(0, x @ model.coefs_[0] + model.intercepts_[0])
    h2 = np.maximum(0, h1 @ model.coefs_[1] + model.intercepts_[1])
    return x, h1, h2


class QuantizedMLP:
    """Uniform symmetric weight/activation quantize-dequantize simulation.

    Biases stay FP32. Percentile ranges use calibration rows only.
    """

    def __init__(self, model: MLPClassifier, calibration: np.ndarray, bits: int):
        qmax = 2 ** (bits - 1) - 1
        self.qmax = qmax
        self.biases = [np.asarray(v, dtype=np.float64) for v in model.intercepts_]
        self.weights = []
        for weights in model.coefs_:
            step = max(float(np.max(np.abs(weights))) / qmax, 1e-12)
            self.weights.append(np.clip(np.rint(weights / step), -qmax, qmax) * step)
        self.ranges = [max(float(np.percentile(np.abs(h), 99.5)), 1e-9)
                       for h in forward_layers(model, calibration)]

    def _round(self, x: np.ndarray, rng: float) -> np.ndarray:
        step = rng / self.qmax
        return np.clip(np.rint(x / step), -self.qmax, self.qmax) * step

    def predict(self, x: np.ndarray, clip_rates: bool = False):
        rates = []
        for layer, weights in enumerate(self.weights):
            if layer < 3:
                rates.append(float(np.mean(np.abs(x) > self.ranges[layer])))
                x = self._round(x, self.ranges[layer])
            x = x @ weights + self.biases[layer]
            if layer < 2:
                x = np.maximum(x, 0)
        probs = expit(x.ravel())
        return (probs, rates) if clip_rates else probs


def float_predict(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def attributions(predict, x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    original = predict(x)
    values = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        altered = x.copy()
        altered[:, j] = reference[j]
        values[:, j] = original - predict(altered)
    return values


def explanation_metrics(fp: np.ndarray, quant: np.ndarray, predict,
                        x: np.ndarray, reference: np.ndarray,
                        random_three: np.ndarray) -> dict[str, np.ndarray]:
    top_fp = np.argsort(-np.abs(fp), axis=1)[:, :3]
    top_q = np.argsort(-np.abs(quant), axis=1)[:, :3]
    overlap = np.array([len(set(a) & set(b)) / 3 for a, b in zip(top_fp, top_q)])
    r_fp = rankdata(np.abs(fp), axis=1)
    r_q = rankdata(np.abs(quant), axis=1)
    aa, bb = r_fp - r_fp.mean(axis=1, keepdims=True), r_q - r_q.mean(axis=1, keepdims=True)
    rho = np.sum(aa * bb, axis=1) / np.maximum(
        np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1), 1e-12
    )
    distance = np.abs(fp - quant).sum(axis=1) / np.maximum(np.abs(fp).sum(axis=1), .01)
    orig = predict(x)
    changed_top, changed_random = x.copy(), x.copy()
    rows = np.arange(len(x))[:, None]
    changed_top[rows, top_q] = reference[top_q]
    changed_random[rows, random_three] = reference[random_three]
    return dict(overlap=overlap, rank_rho=rho, relative_difference=distance,
                top3_effect=np.abs(orig - predict(changed_top)),
                random3_effect=np.abs(orig - predict(changed_random)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="One quick end-to-end run")
    parser.add_argument("--output", type=Path, default=Path("calibration_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x_df, y = load_data(args.output / "cache")
    parts = patient_partitions(y)
    scaler = StandardScaler().fit(x_df.iloc[parts["train"]])
    x = scaler.transform(x_df).astype(np.float64)
    reference = np.median(x[parts["train"]], axis=0)
    rng = np.random.default_rng(444)
    test_pos = parts["test"][y[parts["test"]] == 1]
    test_neg = parts["test"][y[parts["test"]] == 0]
    n_each = 64 if args.pilot else 256
    selected = np.concatenate((rng.choice(test_pos, n_each, replace=False),
                               rng.choice(test_neg, n_each, replace=False)))
    rng.shuffle(selected)
    x_explain, y_explain = x[selected], y[selected]
    random_three = np.stack([rng.choice(len(FEATURES), 3, replace=False)
                             for _ in selected])
    rows = []
    n_seeds, n_draws = (1, 1) if args.pilot else (3, 3)
    print(f"Patients: {len(y)}; positive outcome: {y.mean():.3f}; "
          f"FP32 models: {n_seeds}; calibration cohorts per policy: {n_draws}", flush=True)
    for seed in range(n_seeds):
        # Accuracy-based early stopping predicts only the majority class here,
        # so training stops on log loss, with L2 regularization.
        model = MLPClassifier(hidden_layer_sizes=(32, 16), solver="adam",
                              alpha=1.0, max_iter=90, n_iter_no_change=12,
                              random_state=seed)
        model.fit(x[parts["train"]], y[parts["train"]])
        fp_validation = float_predict(model, x[parts["validation"]])
        # Threshold from FP32 validation predictions, reused for all variants.
        pos_val = np.sort(fp_validation[y[parts["validation"]] == 1])
        threshold = float(np.quantile(pos_val, .30))
        fp_test = float_predict(model, x[parts["test"]])
        fp_explain = attributions(lambda z: float_predict(model, z), x_explain, reference)
        print(f"seed {seed}: FP32 test AUROC={roc_auc_score(y[parts['test']], fp_test):.3f} "
              f"AUPRC={average_precision_score(y[parts['test']], fp_test):.3f}", flush=True)
        for draw in range(n_draws):
            for policy in ("representative", "balanced", "majority_only"):
                crng = np.random.default_rng(10000 + seed * 100 + draw * 10 +
                                              ("representative", "balanced", "majority_only").index(policy))
                picked = cohort(parts["calibration"], y, policy,
                                128 if args.pilot else 256, crng)
                for bits in (8, 4):
                    q = QuantizedMLP(model, x[picked], bits)
                    qt, _ = q.predict(x[parts["test"]], clip_rates=True)
                    qa = attributions(q.predict, x_explain, reference)
                    detail = explanation_metrics(fp_explain, qa, q.predict,
                                                 x_explain, reference, random_three)
                    for label, mask in (("readmitted_within_30_days", y_explain == 1),
                                        ("other", y_explain == 0)):
                        rows.append({
                            "model_seed": seed, "calibration_draw": draw,
                            "policy": policy, "bits": bits, "outcome_group": label,
                            "calibration_positives": int(y[picked].sum()),
                            "test_auroc": roc_auc_score(y[parts["test"]], qt),
                            "test_auprc": average_precision_score(y[parts["test"]], qt),
                            "test_brier": brier_score_loss(y[parts["test"]], qt),
                            "test_sensitivity_at_fixed_threshold": float(np.mean(
                                qt[y[parts["test"]] == 1] >= threshold)),
                            "fp32_test_auroc": roc_auc_score(y[parts["test"]], fp_test),
                            "unchanged_prediction_fraction": float(np.mean(
                                (q.predict(x_explain[mask]) >= threshold)
                                == (float_predict(model, x_explain[mask]) >= threshold))),
                            **{f"mean_{k}": float(np.mean(v[mask])) for k, v in detail.items()},
                            "input_clipped_fraction": float(np.mean(
                                np.abs(x[parts["test"]][y[parts["test"]] == (1 if label == "readmitted_within_30_days" else 0)])
                                > q.ranges[0])),
                        })
                    print(f"  seed={seed} draw={draw} {policy:13s} W{bits}A{bits}: "
                          f"AUROC {rows[-1]['test_auroc']:.3f}; positive top3 overlap "
                          f"{rows[-2]['mean_overlap']:.3f}", flush=True)
    results = pd.DataFrame(rows)
    raw_path = args.output / ("pilot.csv" if args.pilot else "results.csv")
    summary_path = args.output / ("pilot_summary.csv" if args.pilot else "summary.csv")
    results.to_csv(raw_path, index=False)
    summary = results.groupby(["bits", "policy", "outcome_group"], as_index=False).agg(
        variants=("mean_overlap", "size"), auroc=("test_auroc", "mean"),
        auprc=("test_auprc", "mean"), top3_overlap=("mean_overlap", "mean"),
        rank_rho=("mean_rank_rho", "mean"),
        top3_effect=("mean_top3_effect", "mean"),
        random3_effect=("mean_random3_effect", "mean"))
    summary.to_csv(summary_path, index=False)
    print(f"Saved {raw_path} and {summary_path}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Interpretation: simulated quantization; public historical data; "
          "no clinical validation or speed benchmark.")


if __name__ == "__main__":
    main()
