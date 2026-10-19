# German credit: RBF network baseline vs calibrated LightGBM

Script: `analysis.py`. Requirements: `requirements.txt`.

Run `python analysis.py`. It downloads OpenML dataset 31 (German credit), makes a stratified 70/30 split (seed 42), tunes both models by 3-fold CV on the training set only, evaluates once on the test set over seeds 42-44, and writes `results.json` and `figures/baseline_vs_improved.png`.

Metric: macro-F1 on the test set (higher is better).

Fix in this revision: categorical columns are now detected by dtype (non-numeric) rather than `dtype == object`, and nominal byte strings are decoded regardless of pandas string dtype. Previously, on newer pandas, the categorical columns were treated as numeric and the scaler failed on values such as `'0<=X<200'`. The method, models, baseline and split are unchanged.

## Reference

This project is motivated by the following paper:

> E Chumakova, D Korneev, M Gasparian (2024). *The Use of Neural Network Technologies in the Tasks of Controlling IT Risks of a Credit Institution Based on Radial-basis Functions*. Available at SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4785430

Paper author: Prof. Assoc. Prof. Dr. Ekaterina V. Chumakova, Plekhanov Russian University of Economics (chumakova.ev@rea.ru)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
