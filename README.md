# Class-Balanced LightGBM vs RBF Network for Credit Risk Flagging

This experiment is motivated by "The Use of Neural Network Technologies in the Tasks of Controlling IT Risks of a Credit Institution Based on Radial-basis Functions" (2024), which proposes radial-basis function neural networks for monitoring risk in credit institutions. This repository asks whether a gradient-boosted tree model with class balancing, cross-validated tuning and probability calibration can flag risky credit cases better than a standard RBF network.

**Question:** Can a class-balanced, cross-validation-tuned LightGBM model beat a standard radial-basis-function neural network at flagging risky cases on the public German credit dataset?

**Dataset:** German Credit (OpenML id 31, https://www.openml.org/d/31), loaded with the `openml` package. It is a public stand-in for credit-institution risk monitoring, not the paper's own data. The target is good vs bad credit risk. A stratified 70/30 split with seed 42 is used.

**Baseline:** a standard RBF network with k-means centres, Gaussian activations and a logistic-regression output layer. The number of centres and the width are chosen by 3-fold CV on the training data.

**Improved model:** a class-balanced LightGBM with hyperparameters tuned by randomized 3-fold CV on the training data, and sigmoid-calibrated probabilities.

**Metrics:** macro-F1 (main), with risky-class F1, ROC-AUC and log-loss as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and a chart to figures/baseline_vs_improved.png).

## Reference

This project is motivated by the following paper:

> E Chumakova, D Korneev, M Gasparian (2024). *The Use of Neural Network Technologies in the Tasks of Controlling IT Risks of a Credit Institution Based on Radial-basis Functions*. Available at SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4785430

Paper author: Prof. Assoc. Prof. Dr. Ekaterina V. Chumakova, Plekhanov Russian University of Economics (chumakova.ev@rea.ru)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
