# ============================================================
# CatBoost + LightGBM Ensemble (F1) — cleaned & safer defaults
# Key changes vs 0216_0.418.py:
# 1) Multi-select parsing includes commas (matches this dataset delimiter pattern)
# 2) OOF pos-rate constraint OFF by default (often hurts LB if priors differ)
# 3) LGB global FS TOP_K tuned to a more conservative value
# 4) Reduced boilerplate / unused utilities
# ============================================================

import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =====================
# Config (edit here)
# =====================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET = "completed"
ID_COL = "ID"

TEXT_COL = "incumbents_lecture_scale_reason"

CV_SEEDS = [42, 202, 777]
N_SPLITS = 5

# Feature engineering
USE_TEXT_SUMMARY = True
USE_TEXT_EMBEDDING = True
TEXT_EMB_MAX_FEATURES = 4000
TEXT_EMB_DIM = 24

USE_MS_COUNTS = True
USE_MS_TOPTOKENS = True
MS_TOPK = 10
MS_MIN_FREQ = 15
MULTISELECT_COLS_COUNTS = [
    "desired_job", "desired_certificate", "major_field",
    "incumbents_lecture_scale"
]
MULTISELECT_COLS_TOPTOKENS = ["desired_job", "desired_certificate", "major_field"]

USE_RARE_BUCKET = True
RARE_MIN_FREQ = 5

# LGB global feature selection
TOP_K_FEATURES = 100

# Threshold tuning
USE_OOF_POS_CONSTRAINT = False
OOF_POS_MARGIN = 0.05

# Model weighting
W_CB_GRID = np.linspace(0, 1, 101)

# Class-balance controls
USE_CLASS_WEIGHT_CB = False
USE_CLASS_WEIGHT_LGB = True

# Submission sweep controls
EXPORT_RATE_SWEEP = True
RATE_SWEEP = [0.64, 0.66, 0.68, 0.70, 0.72]
REFERENCE_SUBMISSION_FOR_RATE = "0.418.csv"
EXPORT_TEST_PROBS = True


# =====================
# Utils
# =====================
RUN_START = time.time()

def log(msg: str) -> None:
    print(f"[{time.time()-RUN_START:7.1f}s] {msg}", flush=True)

def split_multiselect(x) -> list[str]:
    """Split multi-select labels using common delimiters for this dataset."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    # Treat " / " (space-slash-space) as a delimiter; keep "A/B" intact.
    s = s.replace(" / ", ";")
    parts = re.split(r"[,;/|\n\t]+", s)
    out, seen = [], set()
    for p in parts:
        t = re.sub(r"\s+", " ", p).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def add_text_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].fillna("").astype(str)
    df[f"{col}__len"] = s.str.len()
    df[f"{col}__n_words"] = s.str.split().map(len)
    df[f"{col}__has_digit"] = s.str.contains(r"\d").astype(int)
    df[f"{col}__has_alpha"] = s.str.contains(r"[A-Za-z]").astype(int)
    df[f"{col}__has_kor"] = s.str.contains(r"[가-힣]").astype(int)
    df[f"{col}__has_special"] = s.str.contains(r"[^0-9A-Za-z가-힣\s]").astype(int)
    df[f"{col}__is_empty"] = (s.str.strip() == "").astype(int)
    return df

def fit_text_embedding(train_text: pd.Series, test_text: pd.Series,
                       max_features: int, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=2
    )
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)

    svd = TruncatedSVD(n_components=out_dim, random_state=42)
    Xtr_s = svd.fit_transform(Xtr)
    Xte_s = svd.transform(Xte)
    return Xtr_s, Xte_s

def add_ms_count_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tokens = df[col].apply(split_multiselect)
    df[f"{col}__cnt"] = tokens.map(len)
    df[f"{col}__uniq"] = tokens.map(lambda xs: len(set(xs)))
    return df

def add_ms_toptoken_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             col: str, top_k: int, min_freq: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctr = Counter()
    for xs in train_df[col].apply(split_multiselect):
        ctr.update(xs)
    tokens = [t for t, c in ctr.most_common() if c >= min_freq][:top_k]

    def has_token(series: pd.Series, tok: str) -> pd.Series:
        return series.apply(lambda x: int(tok in set(split_multiselect(x))))

    for tok in tokens:
        safe = re.sub(r"[^0-9A-Za-z가-힣]+", "_", tok).strip("_")[:40]
        fn = f"{col}__has_{safe}"
        train_df[fn] = has_token(train_df[col], tok)
        test_df[fn] = has_token(test_df[col], tok)

    return train_df, test_df

def build_model_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert categorical columns to numeric for LGB: freq + label encoding."""
    tr = train_df.copy()
    te = test_df.copy()

    for c in cat_cols:
        trv = tr[c].astype("string")
        tev = te[c].astype("string")

        vc = trv.value_counts(dropna=False)
        tr[f"{c}__freq"] = trv.map(vc).fillna(0).astype(float)
        te[f"{c}__freq"] = tev.map(vc).fillna(0).astype(float)

        uniq = pd.Index(trv.unique())
        code_map = {v: i for i, v in enumerate(uniq)}
        tr[f"{c}__le"] = trv.map(code_map).fillna(-1).astype(int)
        te[f"{c}__le"] = tev.map(code_map).fillna(-1).astype(int)

    tr = tr.drop(columns=cat_cols)
    te = te.drop(columns=cat_cols)
    return tr, te

def build_splits(y: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out = []
    for seed in CV_SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
            out.append((seed, tr_idx, va_idx))
    return out

def compute_global_importance(X: pd.DataFrame, y: pd.Series, splits, params: dict) -> pd.Series:
    imp = pd.Series(0.0, index=X.columns)
    for seed, tr_idx, va_idx in splits:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        gbm = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            num_boost_round=20000,
            callbacks=[lgb.early_stopping(250, verbose=False)]
        )
        fold_imp = pd.Series(gbm.feature_importance(importance_type="gain"), index=X.columns)
        imp = imp.add(fold_imp, fill_value=0.0)
    return imp

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        t_min=0.01, t_max=0.99, n=981,
                        pos_min=None, pos_max=None) -> tuple[float, float]:
    ts = np.linspace(t_min, t_max, n)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype(int)
        pos = pred.mean()
        if (pos_min is not None and pos < pos_min) or (pos_max is not None and pos > pos_max):
            continue
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)

def tune_weights_threshold(y_true: np.ndarray, oof_cb: np.ndarray, oof_lgb: np.ndarray,
                           w_grid: np.ndarray,
                           pos_min=None, pos_max=None) -> dict:
    best = {"w_cb": 0.5, "t": 0.5, "f1": -1.0}
    for w in w_grid:
        mix = w * oof_cb + (1.0 - w) * oof_lgb
        t, f1 = find_best_threshold(y_true, mix, pos_min=pos_min, pos_max=pos_max)
        if f1 > best["f1"]:
            best = {"w_cb": float(w), "t": float(t), "f1": float(f1)}
    best["w_lgb"] = float(1.0 - best["w_cb"])
    return best

def threshold_for_target_pos_rate(y_prob: np.ndarray, target_pos_rate: float) -> float:
    target_pos_rate = float(np.clip(target_pos_rate, 0.001, 0.999))
    q = float(np.clip(1.0 - target_pos_rate, 0.0, 1.0))
    return float(np.quantile(y_prob, q))

def rate_to_tag(rate: float) -> str:
    return str(round(float(rate), 5)).replace(".", "p")


# =====================
# Main
# =====================
def main():
    log("Load data")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].astype(int)

    # work on a single frame for FE; keep y separate
    X_train = train.drop(columns=[TARGET])
    X_test = test.copy()

    # drop obvious leakage/useless (original code had a fixed list)
    DROP_COLS = [ID_COL, "idea_contest", "contest_award", "contest_participation", "class4"]
    for c in DROP_COLS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c])
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])

    # completed_semester cleanup (kept from original logic)
    if "completed_semester" in X_train.columns:
        for df in (X_train, X_test):
            cs = pd.to_numeric(df["completed_semester"], errors="coerce")
            df["completed_semester__isna"] = cs.isna().astype(int)
            df["completed_semester__gt20"] = (cs > 20).astype(int)
            cs = cs.mask(cs > 20, np.nan)
            df["completed_semester__clean"] = cs
            df.drop(columns=["completed_semester"], inplace=True)

    # code-like numeric -> string categorical (kept)
    CODELIKE = ["school1", "class1", "class2", "class3", "class4", "generation"]
    for c in CODELIKE:
        if c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype("Int64").astype("string").replace({"<NA>": np.nan})
        if c in X_test.columns:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype("Int64").astype("string").replace({"<NA>": np.nan})

    # text features
    if TEXT_COL in X_train.columns:
        if USE_TEXT_SUMMARY:
            log("Text summary features")
            X_train = add_text_summary(X_train, TEXT_COL)
            X_test = add_text_summary(X_test, TEXT_COL)

        if USE_TEXT_EMBEDDING:
            log("Text embedding (TF-IDF char + SVD)")
            tr_text = X_train[TEXT_COL].fillna("").astype(str)
            te_text = X_test[TEXT_COL].fillna("").astype(str)
            tr_svd, te_svd = fit_text_embedding(tr_text, te_text, TEXT_EMB_MAX_FEATURES, TEXT_EMB_DIM)
            for j in range(TEXT_EMB_DIM):
                X_train[f"{TEXT_COL}__svd_{j:02d}"] = tr_svd[:, j]
                X_test[f"{TEXT_COL}__svd_{j:02d}"] = te_svd[:, j]

    # multiselect features
    if USE_MS_COUNTS:
        log("Multi-select count features")
        for c in MULTISELECT_COLS_COUNTS:
            if c in X_train.columns:
                X_train = add_ms_count_features(X_train, c)
                X_test = add_ms_count_features(X_test, c)

    if USE_MS_TOPTOKENS:
        log("Multi-select top-token features")
        for c in MULTISELECT_COLS_TOPTOKENS:
            if c in X_train.columns:
                X_train, X_test = add_ms_toptoken_features(X_train, X_test, c, MS_TOPK, MS_MIN_FREQ)

    # missingness features
    log("Missingness features")
    for df in (X_train, X_test):
        df["row_missing_cnt"] = df.isna().sum(axis=1).astype(int)
        df["row_missing_ratio"] = df.isna().mean(axis=1).astype(float)

    # prepare CatBoost branch
    cat_cols = X_train.select_dtypes(include=["object", "string", "bool", "category"]).columns.tolist()

    X_raw = X_train.copy()
    X_test_raw = X_test.copy()

    for c in cat_cols:
        X_raw[c] = X_raw[c].astype("string").fillna("__MISSING__")
        X_test_raw[c] = X_test_raw[c].astype("string").fillna("__MISSING__")

    if USE_RARE_BUCKET and cat_cols:
        log("Rare bucket for categoricals")
        for c in cat_cols:
            vc = X_raw[c].value_counts()
            rare = set(vc[vc < RARE_MIN_FREQ].index)
            if rare:
                X_raw[c] = X_raw[c].apply(lambda v: "__RARE__" if v in rare else v)
                X_test_raw[c] = X_test_raw[c].apply(lambda v: "__RARE__" if v in rare else v)

    # numeric missing -> median (CatBoost can handle NaN, but filling stabilizes)
    num_cols = X_raw.columns.difference(cat_cols)
    med = X_raw[num_cols].median(numeric_only=True)
    X_raw[num_cols] = X_raw[num_cols].fillna(med)
    X_test_raw[num_cols] = X_test_raw[num_cols].fillna(med)

    # prepare LGB branch (numeric matrix)
    log("Build numeric matrix for LGB")
    X_num, X_test_num = build_model_matrix(X_raw, X_test_raw, cat_cols)

    # weights
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = neg / max(pos, 1.0)
    log(f"class ratio: pos={int(pos)} neg={int(neg)} pos_weight={pos_weight:.3f}")

    splits = build_splits(y)
    n_seed_used = len(set(s for s, _, _ in splits))

    # global FS (LGB)
    log("Global feature importance (LGB)")
    fs_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=48,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=1.0,
        verbose=-1,
        seed=42,
    )
    if USE_CLASS_WEIGHT_LGB:
        fs_params["scale_pos_weight"] = pos_weight

    global_imp = compute_global_importance(X_num, y, splits, fs_params)
    selected = global_imp.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()
    log(f"Selected TOP_K_FEATURES={TOP_K_FEATURES} for LGB")

    X_lgb = X_num[selected]
    X_test_lgb = X_test_num[selected]

    # model params
    cb_params = dict(
        iterations=4000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        od_type="Iter",
        od_wait=200,
        random_strength=1.0,
    )
    if USE_CLASS_WEIGHT_CB:
        cb_params["class_weights"] = [1.0, float(pos_weight)]

    lgb_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=48,
        min_data_in_leaf=30,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=1.0,
        verbose=-1,
        seed=42,
    )
    if USE_CLASS_WEIGHT_LGB:
        lgb_params["scale_pos_weight"] = pos_weight

    # OOF/test containers
    oof_cb = np.zeros(len(X_raw), dtype=float)
    oof_lgb = np.zeros(len(X_raw), dtype=float)
    test_cb = np.zeros(len(X_test_raw), dtype=float)
    test_lgb = np.zeros(len(X_test_raw), dtype=float)

    log("CV train (CatBoost + LGB)")
    for i, (seed, tr_idx, va_idx) in enumerate(splits, 1):
        # CatBoost
        tr_pool = Pool(X_raw.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        va_pool = Pool(X_raw.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        cb = CatBoostClassifier(**{**cb_params, "random_seed": seed})
        cb.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        oof_cb[va_idx] += cb.predict_proba(X_raw.iloc[va_idx])[:, 1] / n_seed_used
        test_cb += cb.predict_proba(X_test_raw)[:, 1] / len(splits)

        # LightGBM
        dtr = lgb.Dataset(X_lgb.iloc[tr_idx], label=y.iloc[tr_idx])
        dva = lgb.Dataset(X_lgb.iloc[va_idx], label=y.iloc[va_idx], reference=dtr)
        gbm = lgb.train(
            lgb_params,
            dtr,
            valid_sets=[dva],
            num_boost_round=20000,
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )
        oof_lgb[va_idx] += gbm.predict(X_lgb.iloc[va_idx], num_iteration=gbm.best_iteration) / n_seed_used
        test_lgb += gbm.predict(X_test_lgb, num_iteration=gbm.best_iteration) / len(splits)

        if i % 5 == 0:
            log(f"fold {i}/{len(splits)} done")

    # threshold constraints (optional)
    pos_rate = y.mean()
    pos_min = pos_rate - OOF_POS_MARGIN if USE_OOF_POS_CONSTRAINT else None
    pos_max = pos_rate + OOF_POS_MARGIN if USE_OOF_POS_CONSTRAINT else None

    # diagnostics
    t_cb, f1_cb = find_best_threshold(y.values, oof_cb, pos_min=pos_min, pos_max=pos_max)
    t_lg, f1_lg = find_best_threshold(y.values, oof_lgb, pos_min=pos_min, pos_max=pos_max)
    best = tune_weights_threshold(y.values, oof_cb, oof_lgb, W_CB_GRID, pos_min=pos_min, pos_max=pos_max)

    log(f"[OOF] CB  best_t={t_cb:.4f} F1={f1_cb:.4f}")
    log(f"[OOF] LGB best_t={t_lg:.4f} F1={f1_lg:.4f}")
    log(f"[OOF] ENS w_cb={best['w_cb']:.2f} w_lgb={best['w_lgb']:.2f} best_t={best['t']:.4f} F1={best['f1']:.4f}")

    # submission
    test_ens = best["w_cb"] * test_cb + best["w_lgb"] * test_lgb
    t_submit = best["t"]
    test_pred = (test_ens >= t_submit).astype(int)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    out_path = "submission_cb_lgb_msfix.csv"
    sub.to_csv(out_path, index=False)

    log(f"Saved: {out_path} | test_pos_rate={test_pred.mean():.4f} | t_submit={t_submit:.4f}")

    # Save probabilities once, so threshold sweeps can be done without retraining.
    if EXPORT_TEST_PROBS:
        prob_path = "submission_cb_lgb_msfix_probs.csv"
        pd.DataFrame({ID_COL: test[ID_COL], "prob": test_ens}).to_csv(prob_path, index=False)
        log(f"Saved: {prob_path}")

    # Additional submissions by target positive rate
    if EXPORT_RATE_SWEEP:
        rate_targets = list(RATE_SWEEP)
        if REFERENCE_SUBMISSION_FOR_RATE:
            try:
                ref_df = pd.read_csv(REFERENCE_SUBMISSION_FOR_RATE)
                if TARGET in ref_df.columns:
                    ref_rate = float(ref_df[TARGET].astype(int).mean())
                    if all(abs(ref_rate - x) > 1e-9 for x in rate_targets):
                        rate_targets.append(ref_rate)
                    log(f"Reference rate from {REFERENCE_SUBMISSION_FOR_RATE}: {ref_rate:.5f}")
            except Exception as e:
                log(f"Reference rate read skipped: {e}")

        for rate in sorted(rate_targets):
            t_rate = threshold_for_target_pos_rate(test_ens, rate)
            pred_rate = (test_ens >= t_rate).astype(int)
            out_rate = f"submission_cb_lgb_msfix_rate{rate_to_tag(rate)}.csv"
            pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_rate}).to_csv(out_rate, index=False)
            log(
                f"Saved: {out_rate} | target_rate={rate:.5f} "
                f"| actual_rate={pred_rate.mean():.5f} | t={t_rate:.5f}"
            )

if __name__ == "__main__":
    main()
