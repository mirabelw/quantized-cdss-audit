# Auditing Quantization in a Clinical Prediction Testbed

Code and derived results for the paper *Auditing Quantization in a Clinical
Prediction Testbed: Explanation Stability and Calibration-Set Privacy*
(anonymized for review).

The study fits three small MLPs to predict 30-day readmission on the public
UCI Diabetes 130-US Hospitals dataset, then measures how post-training
quantization changes (1) field-level explanations and fixed-threshold
decisions and (2) what the saved MinMax activation scales reveal about the
patients used for calibration. It is a technical testbed, not a clinical tool.

## Data

The scripts download the dataset automatically from the UCI Machine Learning
Repository on first run (no account needed) and cache it locally:
<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
(DOI 10.24432/C5230J, CC BY 4.0). Raw patient records are not included here.

## Setup

Python 3.12, CPU only.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1
```

Native results in the paper used ONNX Runtime 1.30.0 (versions recorded in
`native_matmul_results/native_runtime_versions.json`). Other ONNX Runtime
versions may give slightly different native INT8 numbers.

## Reproduce the paper

Run from the repository root, in this order (the first command downloads the
data into `repaired_cohort_results/cache/`, which the later scripts reuse).

| Paper item | Command | Output |
|---|---|---|
| Table I, Fig. 1, paired intervals, cohort-policy contrasts | `python repaired_cohort_study.py` then `python analyze_repaired.py` and `python make_repaired_figure.py` | `repaired_cohort_results/` |
| Simulated scale matching (150 draws per seed and size) | `python validate_stronger_privacy.py` | `repaired_cohort_results/stronger_privacy_*.csv` |
| Table II (native INT8 explanations) and Table III (native privacy flags) | `python validate_stronger_onnx.py --draws 12 --matmul-only --output native_matmul_results` | `native_matmul_results/` |
| Matched pooled-MinMax W8A8 simulation | `python repaired_cohort_study.py --only-config W8A8_pooled_minmax --output repaired_minmax_results` | `repaired_minmax_results/` |
| Full-graph QDQ risk-score diagnostic (28–65 distinct values) | `python validate_stronger_onnx.py --draws 1 --output fullgraph_grid_pilot` | `fullgraph_grid_pilot/` |
| Scales and privacy flags under full-graph QDQ (144-draw comparison) | `python validate_stronger_onnx.py --draws 12 --output fullgraph_qdq_results` | `fullgraph_qdq_results/` |
| Convergence sensitivity analysis (cap 250) | `python repaired_cohort_study.py --max-iter 250 --output repaired_convergence_results` | `repaired_convergence_results/` |

The primary fits stop at the pre-specified 90-iteration cap and emit
scikit-learn convergence warnings; these are expected and disclosed in the
paper.

## Files

- `repaired_cohort_study.py`: data preparation, model fitting, simulated
  quantization, field-ablation explanations, and utility metrics.
- `analyze_repaired.py`: paired contrasts and three-seed intervals.
- `make_repaired_figure.py`: Fig. 1.
- `validate_stronger_privacy.py`: simulated MinMax and exact-percentile
  scale matching.
- `validate_stronger_onnx.py`: ONNX export, native ONNX Runtime INT8
  calibration (MinMax and histogram Percentile), scale extraction, native
  explanation audit, and paired simulation checks.
- `calibration_explanations.py`: shared data loader and helper functions
  imported by the scripts above (its own command-line entry point is an
  earlier nine-feature pilot that the paper does not use).
- `native_matmul_results/native_strong_fp32_31{1,2,3}.onnx`: the three
  exported full-precision models.

## Notes on interpretation

- The privacy test is white-box: it assumes the attacker has the candidate
  records, the exact fitted preprocessing (not stored in the ONNX file), and
  the full-precision parent network. Member and nonmember candidates are
  balanced 50:50 by construction, so flag precision is not a real-world
  positive predictive value.
- Repeated calibration draws reuse patients from the same candidate pool;
  pooled counts are not independent patient-level trials.
- Four-bit results are arithmetic simulations; only INT8 was run natively.
- In `fullgraph_qdq_results/`, only the scale and privacy columns are used in
  the paper. Its explanation and utility columns include quantization of the
  probability output and are not comparable with the simulator.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, from the UCI Machine Learning
Repository.
