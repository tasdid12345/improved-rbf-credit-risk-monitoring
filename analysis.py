"""Baseline (k-means RBF network) vs improved (class-balanced, CV-tuned, calibrated LightGBM)
on the German credit data (OpenML id 31). Writes results.json and figures/."""
import json, os, sys, tempfile, time

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openml
import lightgbm as lgb
from scipy.io import arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # model randomness only; the data split is fixed at SEED
DATA_ID = 31
TEST_SIZE = 0.3


def with_retries(fn, tries=5):
    """Call fn(); retry with exponential backoff on connection errors, HTTP 429 and 5xx."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            retryable = code is None or code == 429 or code >= 500
            if i == tries - 1 or not retryable:
                raise
            print(f"  request failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def fetch(url):
    """Download bytes with a 60 s timeout; raise HTTPError on bad status codes."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def is_categorical(s):
    """Nominal/string columns are categorical; anything numeric is not."""
    return not pd.api.types.is_numeric_dtype(s)


def load_data():
    """Load OpenML dataset 31: metadata via the openml package, ARFF file via requests."""
    print("Loading German credit (OpenML id 31) ...")
    ds = with_retries(lambda: openml.datasets.get_dataset(
        DATA_ID, download_data=False, download_qualities=False))
    raw = with_retries(lambda: fetch(ds.url))
    with tempfile.NamedTemporaryFile(suffix=".arff", delete=False) as f:
        f.write(raw)
        path = f.name
    data, _ = arff.loadarff(path)
    os.remove(path)
    df = pd.DataFrame(data)
    for c in df.columns:  # decode nominal byte strings (works for object or pandas string dtypes)
        if is_categorical(df[c]):
            df[c] = pd.Series(
                [v.decode() if isinstance(v, bytes) else v for v in df[c].tolist()],
                index=df.index, dtype=object)
    y = (df.pop("class") == "bad").astype(int).to_numpy()  # positive class = risky ("bad")
    return df, y


def make_preprocessor(X_train):
    """Standardise numeric columns and one-hot encode categoricals (fit later on train folds only)."""
    cat = [c for c in X_train.columns if is_categorical(X_train[c])]
    num = [c for c in X_train.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


class RBFNet(BaseEstimator, ClassifierMixin):
    """Gaussian RBF network: k-means centres, Broomhead-Lowe width, logistic-regression output."""

    def __init__(self, n_centers=20, width_scale=1.0, seed=42):
        self.n_centers = n_centers
        self.width_scale = width_scale
        self.seed = seed

    def _phi(self, X):
        d2 = ((X[:, None, :] - self.centers_[None]) ** 2).sum(-1)  # squared distances to centres
        return np.exp(-d2 / (2 * self.sigma_ ** 2))

    def fit(self, X, y):
        self.centers_ = KMeans(self.n_centers, n_init=5, random_state=self.seed).fit(X).cluster_centers_
        dmax = np.max(np.linalg.norm(self.centers_[:, None] - self.centers_[None], axis=-1))
        self.sigma_ = self.width_scale * dmax / np.sqrt(2 * self.n_centers)
        self.out_ = LogisticRegression(max_iter=1000).fit(self._phi(X), y)
        return self

    def predict_proba(self, X):
        return self.out_.predict_proba(self._phi(X))

    def predict(self, X):
        return self.out_.predict(self._phi(X))


def evaluate(model, X, y):
    """Shared evaluation: macro-F1 (main), risky-class F1, ROC-AUC and log-loss on the test set."""
    proba = model.predict_proba(X)
    pred = model.predict(X)
    return {"macro_f1": f1_score(y, pred, average="macro"), "f1": f1_score(y, pred),
            "auc": roc_auc_score(y, proba[:, 1]), "logloss": log_loss(y, proba)}


def main():
    t0 = time.time()
    np.random.seed(SEED)
    df, y = load_data()
    print(f"Rows: {len(df)}, raw features: {df.shape[1]}")
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train {len(y_tr)} / test {len(y_te)} (stratified split, seed {SEED})")
    pre = make_preprocessor(X_tr)

    base_runs, imp_runs, notes = [], [], []
    for seed in SEEDS:
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)  # same folds for both models

        print(f"[seed {seed}] baseline: tuning RBF net (centres, width) by 3-fold CV ...")
        base_pipe = Pipeline([("prep", pre), ("rbf", RBFNet(seed=seed))])
        base = GridSearchCV(base_pipe,
                            {"rbf__n_centers": [10, 25, 50], "rbf__width_scale": [0.5, 1.0, 2.0]},
                            scoring="f1_macro", cv=cv).fit(X_tr, y_tr)
        base_runs.append(evaluate(base.best_estimator_, X_te, y_te))

        print(f"[seed {seed}] improved: tuning class-balanced LightGBM by 3-fold CV ...")
        lgb_pipe = Pipeline([("prep", pre), ("lgbm", lgb.LGBMClassifier(
            class_weight="balanced", random_state=seed, verbose=-1))])
        grid = {"lgbm__n_estimators": [100, 200, 400], "lgbm__learning_rate": [0.02, 0.05, 0.1],
                "lgbm__num_leaves": [7, 15, 31], "lgbm__min_child_samples": [5, 10, 20, 40],
                "lgbm__reg_lambda": [0.0, 1.0, 5.0]}
        tuned = RandomizedSearchCV(lgb_pipe, grid, n_iter=12, scoring="f1_macro",
                                   cv=cv, random_state=seed).fit(X_tr, y_tr)
        print(f"[seed {seed}] improved: calibrating probabilities (sigmoid, 3-fold) ...")
        calib = CalibratedClassifierCV(tuned.best_estimator_, method="sigmoid", cv=3).fit(X_tr, y_tr)
        imp_runs.append(evaluate(calib, X_te, y_te))

        notes.append(f"seed {seed}: RBF best {base.best_params_}; LightGBM best {tuned.best_params_}")
        print(f"  baseline macro-F1 {base_runs[-1]['macro_f1']:.4f} | "
              f"improved macro-F1 {imp_runs[-1]['macro_f1']:.4f}")

    def agg(runs, key):
        v = [r[key] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    b_f1, b_f1_sd = agg(base_runs, "macro_f1")
    i_f1, i_f1_sd = agg(imp_runs, "macro_f1")
    metrics = {
        "baseline_score": b_f1, "improved_score": i_f1, "improvement": i_f1 - b_f1,
        "baseline_score_std": b_f1_sd, "improved_score_std": i_f1_sd,
        "baseline_f1": agg(base_runs, "f1")[0], "improved_f1": agg(imp_runs, "f1")[0],
        "baseline_auc": agg(base_runs, "auc")[0], "improved_auc": agg(imp_runs, "auc")[0],
        "baseline_logloss": agg(base_runs, "logloss")[0], "improved_logloss": agg(imp_runs, "logloss")[0],
    }
    if not all(np.isfinite(list(metrics.values()))):
        print("Non-finite metrics; cannot produce meaningful results.")
        sys.exit(1)

    results = {
        "question": "Can a class-balanced, cross-validation-tuned LightGBM model beat a standard "
                    "radial-basis-function neural network at flagging risky cases on the public German credit dataset?",
        "method": "Stratified 70/30 split (seed 42); preprocessing fit on train only; baseline RBF net and "
                  "improved LightGBM tuned by 3-fold CV on train; test set used once; seeds 42-44.",
        "dataset": {"name": "German Credit (OpenML id 31)", "source_url": "https://www.openml.org/d/31",
                    "n_rows": int(len(df)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "macro-F1 (test set)",
        "higher_is_better": True,
        "baseline": "Standard RBF network: k-means centres, Gaussian RBF activations, logistic-regression "
                    "output; number of centres and width chosen by CV on train.",
        "improved": "Class-balanced LightGBM with hyperparameters tuned by randomized CV search on train "
                    "and sigmoid-calibrated probabilities.",
        "metrics": metrics,
        "notes": "German credit is a stand-in for the paper's IT-risk data, not the paper's own dataset. "
                 "Predictions use the default 0.5 threshold (argmax). The test set has only ~300 rows, so "
                 "differences are noisy. Seeds vary model randomness only; the split is fixed. "
                 "Secondary metrics are seed means. " + " | ".join(notes),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Bar chart: macro-F1 and ROC-AUC, baseline vs improved (means over seeds)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [b_f1, metrics["baseline_auc"]], 0.4, label="Baseline RBF net")
    ax.bar(x + 0.2, [i_f1, metrics["improved_auc"]], 0.4, label="Improved LightGBM")
    ax.set_xticks(x, ["macro-F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("German credit test set (mean over seeds)")
    fig.savefig("figures/baseline_vs_improved.png", dpi=120, bbox_inches="tight")
    print(f"Done in {time.time() - t0:.0f}s. Results written to results.json")


if __name__ == "__main__":
    main()
