# German credit: RBF network baseline vs calibrated LightGBM

Script: `analysis.py`. Requirements: `requirements.txt`.

Run `python analysis.py`. It downloads OpenML dataset 31 (German credit), makes a stratified 70/30 split (seed 42), tunes both models by 3-fold CV on the training set only, evaluates once on the test set over seeds 42-44, and writes `results.json` and `figures/baseline_vs_improved.png`.

Metric: macro-F1 on the test set (higher is better).

Fix in this revision: categorical columns are now detected by dtype (non-numeric) rather than `dtype == object`, and nominal byte strings are decoded regardless of pandas string dtype. Previously, on newer pandas, the categorical columns were treated as numeric and the scaler failed on values such as `'0<=X<200'`. The method, models, baseline and split are unchanged.

## Results (latest run)

On the public German Credit dataset (OpenML 31), a class-balanced, CV-tuned LightGBM reached 64% macro-F1 on a single 300-row test split, versus 41% for a standard k-means radial-basis-function network, a gain of about 23 points.

| Metric | Value |
|---|---|
| baseline_score | 0.4118 |
| improved_score | 0.6414 |
| improvement | 0.2296 |
| baseline_score_std | 5.551e-17 |
| improved_score_std | 0.01755 |
| baseline_f1 | 0 |
| improved_f1 | 0.4412 |
| baseline_auc | 0.6673 |
| improved_auc | 0.7848 |
| baseline_logloss | 0.5849 |
| improved_logloss | 0.505 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: German credit is a stand-in for the paper's IT-risk data, not the paper's own dataset. Predictions use the default 0.5 threshold (argmax). The test set has only ~300 rows, so differences are noisy. Seeds vary model randomness only; the split is fixed. Secondary metrics are seed means. seed 42: RBF best {'rbf__n_centers': 10, 'rbf__width_scale': 2.0}; LightGBM best {'lgbm__reg_lambda': 0.0, 'lgbm__num_leaves': 7, 'lgbm__n_estimators': 200, 'lgbm__min_child_samples': 10, 'lgbm__learning_rate': 0.05} | seed 43: RBF best {'rbf__n_centers': 10, 'rbf__width_scale': 0.5}; LightGBM best {'lgbm__reg_lambda': 1.0, 'lgbm__num_leaves': 31, 'lgbm__n_estimators': 100, 'lgbm__min_child_samples': 5, 'lgbm__learning_rate': 0.05} | seed 44: RBF best {'rbf__n_centers': 10, 'rbf__width_scale': 2.0}; LightGBM 

## Reference

This project is motivated by the following paper:

> E Chumakova, D Korneev, M Gasparian (2024). *The Use of Neural Network Technologies in the Tasks of Controlling IT Risks of a Credit Institution Based on Radial-basis Functions*. Available at SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4785430

Paper author: Prof. Assoc. Prof. Dr. Ekaterina V. Chumakova, Plekhanov Russian University of Economics (chumakova.ev@rea.ru)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
