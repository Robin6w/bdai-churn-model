# ============================================================
# Hybrid Stable Pipeline: Global FS + CatBoost/LGB Ensemble
# - base flow from 0216.py (stable)
# - model core switched to CatBoost + LightGBM
# - train-fit/test-transform text embedding
# - multi-select count + train-based top-token features
# - tuned-threshold submission (no forced pos-rate match by default)
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
# Config
# =====================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET = "completed"
ID_COL = "ID"
TEXT_COL = "incumbents_lecture_scale_reason"

LIGHT_MODE = False
CV_SEEDS = [42, 202] if LIGHT_MODE else [42, 202, 777]
N_SPLITS = 5

# Feature engineering
USE_ID_FEATURES = False
USE_MS_COUNTS = True
USE_MS_TOPTOKENS = True
MS_TOPK = 10 if not LIGHT_MODE else 6
MS_MIN_FREQ = 15 if not LIGHT_MODE else 10

USE_TEXT_SUMMARY = True
USE_TEXT_EMBEDDING = True
TEXT_EMB_MAX_FEATURES = 4000 if not LIGHT_MODE else 2000
TEXT_EMB_DIM = 24 if not LIGHT_MODE else 12

USE_RARE_BUCKET = True
RARE_MIN_FREQ = 5

# Feature selection
TOP_K_FEATURES = 80 if not LIGHT_MODE else 60
TOP_K_CANDIDATES = [60, 80, 100] if not LIGHT_MODE else [40, 55, 70]
RUN_TOPK_ABLATION = True

# Threshold / submission
SUBMIT_USE_TUNED_THRESHOLD = True
TARGET_POS_RATE_SINGLE = 0.30
USE_OOF_POS_CONSTRAINT = True
OOF_POS_MARGIN = 0.12
USE_SUBMISSION_POS_GUARD = False  # user requested no fixed pos-rate

# Model weighting
W_CB_GRID = np.linspace(0, 1, 101)

# Class-balance controls
USE_CLASS_WEIGHT_CB = False
USE_CLASS_WEIGHT_LGB = True

RUN_START = time.time()


def log_step(msg):
    elapsed = time.time() - RUN_START
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def threshold_for_target_pos_rate(y_prob, target_pos_rate):
    target_pos_rate = float(np.clip(target_pos_rate, 0.001, 0.999))
    q = float(np.clip(1.0 - target_pos_rate, 0.0, 1.0))
    return float(np.quantile(y_prob, q))


# comma included
def split_multiselect(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    parts = re.split(r"[,;/|\n\t]+", s)
    out, seen = [], set()
    for p in parts:
        t = re.sub(r"\s+", " ", p).strip()
        if t and (t not in seen):
            seen.add(t)
            out.append(t)
    return out


def add_text_summary(df, col):
    s = df[col].fillna("").astype(str)
    df[f"{col}__len"] = s.str.len().astype(np.int32)
    df[f"{col}__word_cnt"] = s.apply(lambda x: len(x.split())).astype(np.int16)
    df[f"{col}__has_digit"] = s.str.contains(r"\d", regex=True).astype(np.int8)
    df[f"{col}__has_alpha"] = s.str.contains(r"[A-Za-z]", regex=True).astype(np.int8)
    df[f"{col}__has_special"] = s.str.contains(r"[^0-9A-Za-z가-힣\s]", regex=True).astype(np.int8)
    df[f"{col}__is_empty"] = (s.str.strip() == "").astype(np.int8)
    return df


# train-fit / test-transform only
def fit_text_embedding(train_text, test_text, max_features, n_components, random_state=42):
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)

    if Xtr.shape[1] <= 1:
        return None, None, 0

    k = min(n_components, Xtr.shape[1] - 1)
    if k <= 0:
        return None, None, 0

    svd = TruncatedSVD(n_components=k, random_state=random_state)
    Xtr_svd = svd.fit_transform(Xtr).astype(np.float32)
    Xte_svd = svd.transform(Xte).astype(np.float32)
    return Xtr_svd, Xte_svd, k


def add_ms_count_features(df, col):
    toks = df[col].apply(split_multiselect)
    df[f"{col}__cnt"] = toks.apply(len).astype(np.int16)
    df[f"{col}__uniq"] = toks.apply(lambda xs: len(set(xs))).astype(np.int16)
    return df


# train-only token selection
def add_ms_toptoken_features(train_df, test_df, col, top_k=10, min_freq=10):
    tr_toks = train_df[col].apply(split_multiselect)
    te_toks = test_df[col].apply(split_multiselect)

    freq = Counter()
    for xs in tr_toks:
        freq.update(xs)

    top_tokens = [t for t, c in freq.most_common() if c >= min_freq][:top_k]
    created = []
    if not top_tokens:
        return train_df, test_df, created

    for t in top_tokens:
        safe = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", t)[:40]
        f = f"{col}__has_{safe}"
        created.append(f)
        train_df[f] = tr_toks.apply(lambda xs: int(t in xs)).astype(np.int8)
        test_df[f] = te_toks.apply(lambda xs: int(t in xs)).astype(np.int8)

    return train_df, test_df, created


def build_model_matrix(train_df, test_df, cat_cols):
    tr = train_df.copy()
    te = test_df.copy()
    for c in cat_cols:
        vc = tr[c].value_counts(dropna=False)
        tr[c + "__freq"] = tr[c].map(vc).fillna(0).astype(np.float32)
        te[c + "__freq"] = te[c].map(vc).fillna(0).astype(np.float32)

        allv = pd.concat([tr[c], te[c]], axis=0).astype("string")
        codes, _ = pd.factorize(allv, sort=False)
        tr[c + "__le"] = codes[: len(tr)].astype(np.int32)
        te[c + "__le"] = codes[len(tr):].astype(np.int32)

    tr = tr.drop(columns=cat_cols)
    te = te.drop(columns=cat_cols)
    return tr, te


def find_best_threshold(y_true, y_prob, t_min=0.01, t_max=0.99, n=981, min_pos_rate=None, max_pos_rate=None):
    ts = np.linspace(t_min, t_max, n)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype(int)
        pos_rate = float(pred.mean())
        if (min_pos_rate is not None) and (pos_rate < min_pos_rate):
            continue
        if (max_pos_rate is not None) and (pos_rate > max_pos_rate):
            continue
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    if best_f1 < 0:
        for t in ts:
            pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def tune_weights_threshold(y_true, oof_cb, oof_lgb, w_grid):
    pos_rate = float(np.mean(y_true))
    min_pos = max(0.01, pos_rate - OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None
    max_pos = min(0.99, pos_rate + OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None

    best = None
    total = len(w_grid)
    log_every = max(1, total // 20)

    for i, w_cb in enumerate(w_grid, 1):
        mix = w_cb * oof_cb + (1.0 - w_cb) * oof_lgb
        t, f1 = find_best_threshold(y_true, mix, n=981, min_pos_rate=min_pos, max_pos_rate=max_pos)
        if (best is None) or (f1 > best["f1"]):
            best = {
                "w_cb": float(w_cb),
                "w_lgb": float(1.0 - w_cb),
                "t": float(t),
                "f1": float(f1),
            }
            print(
                f"[Tune] best_update {i}/{total} | w_cb={best['w_cb']:.3f} "
                f"| w_lgb={best['w_lgb']:.3f} | t={best['t']:.5f} | f1={best['f1']:.5f}",
                flush=True,
            )
        if (i % log_every == 0) or (i == total):
            print(f"[Tune] progress {i}/{total} ({100*i/total:.1f}%)", flush=True)

    return best


def build_splits(X_any, y):
    splits = []
    for seed in CV_SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X_any, y):
            splits.append((seed, tr_idx, va_idx))
    return splits


def compute_global_importance(X_num, y, splits, fs_params):
    gain_sum = np.zeros(X_num.shape[1], dtype=float)
    t0 = time.time()
    for i, (_, tr_idx, va_idx) in enumerate(splits, 1):
        dtr = lgb.Dataset(X_num.iloc[tr_idx], label=y.iloc[tr_idx])
        dva = lgb.Dataset(X_num.iloc[va_idx], label=y.iloc[va_idx])
        booster = lgb.train(
            fs_params,
            dtr,
            num_boost_round=2000 if not LIGHT_MODE else 1200,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        )
        gain_sum += booster.feature_importance(importance_type="gain")
        if i % 5 == 0 or i == len(splits):
            log_step(f"[FS-global] fold {i}/{len(splits)} | elapsed={(time.time()-t0)/60:.1f}m")
    return gain_sum


# =====================
# Load
# =====================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
y = train[TARGET].astype(int).copy()
log_step(f"Loaded: train={train.shape}, test={test.shape}, pos_rate={y.mean():.3f}")

train_n = len(train)
train_text_raw = train[TEXT_COL].fillna("").astype(str) if TEXT_COL in train.columns else pd.Series([""] * len(train))
test_text_raw = test[TEXT_COL].fillna("").astype(str) if TEXT_COL in test.columns else pd.Series([""] * len(test))

# =====================
# Feature engineering
# =====================
all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

for c in list(all_df.columns):
    if c == ID_COL:
        continue
    if all_df[c].nunique(dropna=False) <= 1:
        all_df.drop(columns=[c], inplace=True)

for c in ["idea_contest", "contest_award", "contest_participation", "class4", ID_COL]:
    if c in all_df.columns:
        all_df.drop(columns=[c], inplace=True)

if "completed_semester" in all_df.columns:
    cs = pd.to_numeric(all_df["completed_semester"], errors="coerce")
    all_df["completed_semester__is_big"] = (cs > 20).astype(np.int8)
    all_df["completed_semester__isna"] = cs.isna().astype(np.int8)
    all_df["completed_semester__clean"] = cs.where(cs <= 20, np.nan).astype(np.float32)
    all_df.drop(columns=["completed_semester"], inplace=True)

for c in ["school1", "class1", "class2", "class3", "class4", "generation"]:
    if c in all_df.columns:
        all_df[c] = all_df[c].astype("Int64").astype("string").replace({"<NA>": np.nan})

if USE_ID_FEATURES and (ID_COL in train.columns):
    id_all = pd.concat([train[ID_COL], test[ID_COL]], axis=0, ignore_index=True).astype(str)
    id_num = pd.to_numeric(id_all.str.extract(r"(\d+)$")[0], errors="coerce")
    all_df["id_num"] = id_num.astype(np.float32)
    all_df["id_rank_pct"] = id_num.rank(method="average", pct=True).astype(np.float32)
    all_df["id_num_mod10"] = id_num.fillna(-1).astype(np.int64).mod(10).astype(np.int8)

if USE_TEXT_SUMMARY:
    tmp_all = pd.concat([train_text_raw, test_text_raw], axis=0, ignore_index=True)
    tmp_df = pd.DataFrame({TEXT_COL: tmp_all})
    tmp_df = add_text_summary(tmp_df, TEXT_COL)
    for cc in tmp_df.columns:
        if cc != TEXT_COL:
            all_df[cc] = tmp_df[cc].values

if USE_TEXT_EMBEDDING:
    Xtr_svd, Xte_svd, k = fit_text_embedding(
        train_text=train_text_raw,
        test_text=test_text_raw,
        max_features=TEXT_EMB_MAX_FEATURES,
        n_components=TEXT_EMB_DIM,
        random_state=42,
    )
    if k > 0:
        for j in range(k):
            all_df[f"{TEXT_COL}__svd_{j:02d}"] = np.concatenate([Xtr_svd[:, j], Xte_svd[:, j]], axis=0)
        log_step(f"Text embedding added: svd_dim={k}")

ms_split_and_cnt = ["desired_certificate", "major_field", "desired_job"]
ms_cnt_only = ["expected_domain", "desired_job_except_data", "onedayclass_topic", "interested_company", "previous_class_8"]

if USE_MS_COUNTS:
    for c in (ms_split_and_cnt + ms_cnt_only):
        if c in all_df.columns:
            all_df = add_ms_count_features(all_df, c)

if USE_MS_TOPTOKENS:
    X_tmp = all_df.iloc[:train_n].copy()
    X_te_tmp = all_df.iloc[train_n:].copy()
    created_all = []
    for c in ms_split_and_cnt:
        if c in X_tmp.columns:
            X_tmp, X_te_tmp, created = add_ms_toptoken_features(
                X_tmp, X_te_tmp, c, top_k=MS_TOPK, min_freq=MS_MIN_FREQ
            )
            created_all += created
    all_df = pd.concat([X_tmp, X_te_tmp], axis=0, ignore_index=True)
    if created_all:
        log_step(f"MS top-token features added: {len(created_all)}")

row_missing = all_df.isna().sum(axis=1)
all_df["row_missing_cnt"] = row_missing.astype(np.int16)
all_df["row_missing_ratio"] = (row_missing / max(1, all_df.shape[1])).astype(np.float32)
for c in all_df.select_dtypes(include=["number"]).columns:
    if all_df[c].isna().any():
        all_df[c + "__isna"] = all_df[c].isna().astype(np.int8)

# =====================
# Split + matrix preparation
# =====================
X_raw = all_df.iloc[:train_n].copy()
X_test_raw = all_df.iloc[train_n:].copy()

cat_cols = X_raw.select_dtypes(include=["object", "string", "category"]).columns.tolist()
for c in cat_cols:
    X_raw[c] = X_raw[c].astype("string").fillna("__MISSING__")
    X_test_raw[c] = X_test_raw[c].astype("string").fillna("__MISSING__")

if USE_RARE_BUCKET and cat_cols:
    for c in cat_cols:
        vc = X_raw[c].value_counts(dropna=False)
        keep = set(vc[vc >= RARE_MIN_FREQ].index)
        X_raw[c] = X_raw[c].where(X_raw[c].isin(keep), "__RARE__")
        X_test_raw[c] = X_test_raw[c].where(X_test_raw[c].isin(keep), "__RARE__")

num_cols_raw = X_raw.select_dtypes(include=["number"]).columns.tolist()
if num_cols_raw:
    med_raw = X_raw[num_cols_raw].median(numeric_only=True)
    X_raw[num_cols_raw] = X_raw[num_cols_raw].fillna(med_raw)
    X_test_raw[num_cols_raw] = X_test_raw[num_cols_raw].fillna(med_raw)

X_num, X_test_num = build_model_matrix(X_raw, X_test_raw, cat_cols)
feature_names = X_num.columns.tolist()
log_step(f"Feature matrix ready: X_raw={X_raw.shape}, X_num={X_num.shape}, X_test={X_test_num.shape}, n_cat={len(cat_cols)}")

splits = build_splits(X_num, y)
log_step(f"Prepared CV splits: total_folds={len(splits)}")

# =====================
# Global feature selection (for LGB branch)
# =====================
pos = int(y.sum())
neg = int(len(y) - pos)
pos_weight = neg / max(pos, 1)

fs_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=48 if not LIGHT_MODE else 40,
    min_data_in_leaf=30,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=2.0,
    scale_pos_weight=pos_weight if USE_CLASS_WEIGHT_LGB else 1.0,
    verbose=-1,
)

if RUN_TOPK_ABLATION:
    log_step(f"Start TOP-K ablation: {TOP_K_CANDIDATES}")
    gain_sum = compute_global_importance(X_num, y, splits, fs_params)
    imp_df = pd.DataFrame({"feature": feature_names, "gain": gain_sum}).sort_values("gain", ascending=False)

    best_k = TOP_K_FEATURES
    best_k_f1 = -1.0
    n_seed_used = len({s for s, _, _ in splits})

    for k in TOP_K_CANDIDATES:
        feats = imp_df.head(int(k))["feature"].tolist()
        Xk = X_num[feats]

        oof_lgb_k = np.zeros(len(Xk), dtype=float)
        for _, tr_idx, va_idx in splits:
            X_tr = Xk.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_va = Xk.iloc[va_idx]
            y_va = y.iloc[va_idx]

            dtr = lgb.Dataset(X_tr, label=y_tr)
            dva = lgb.Dataset(X_va, label=y_va)
            booster = lgb.train(
                fs_params,
                dtr,
                num_boost_round=1500 if not LIGHT_MODE else 900,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
            )
            oof_lgb_k[va_idx] += booster.predict(X_va, num_iteration=booster.best_iteration) / n_seed_used

        min_pos = max(0.01, float(y.mean()) - OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None
        max_pos = min(0.99, float(y.mean()) + OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None
        _, f1_k = find_best_threshold(y, oof_lgb_k, n=981, min_pos_rate=min_pos, max_pos_rate=max_pos)
        log_step(f"[Ablation] k={k} -> LGB OOF F1={f1_k:.5f}")

        if f1_k > best_k_f1:
            best_k_f1 = f1_k
            best_k = int(k)

    TOP_K_FEATURES = best_k
    log_step(f"Ablation chose TOP_K_FEATURES={TOP_K_FEATURES} (best_lgb_oof_f1={best_k_f1:.5f})")

log_step("Compute final global feature importance...")
gain_sum = compute_global_importance(X_num, y, splits, fs_params)
imp_df = pd.DataFrame({"feature": feature_names, "gain": gain_sum}).sort_values("gain", ascending=False)
selected_features = imp_df.head(int(TOP_K_FEATURES))["feature"].tolist()
log_step(f"Selected features for LGB: {len(selected_features)} / {len(feature_names)}")

X_lgb = X_num[selected_features].copy()
X_lgb_test = X_test_num[selected_features].copy()

# =====================
# Train CatBoost + LightGBM (repeated CV)
# =====================
cb_params = dict(
    iterations=4000 if not LIGHT_MODE else 2500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    l2_leaf_reg=6,
    random_strength=1.0,
    od_type="Iter",
    od_wait=200,
)
if USE_CLASS_WEIGHT_CB:
    cb_params["class_weights"] = [1.0, float(pos_weight)]

lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=48 if not LIGHT_MODE else 40,
    min_data_in_leaf=30,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=2.0,
    scale_pos_weight=pos_weight if USE_CLASS_WEIGHT_LGB else 1.0,
    verbose=-1,
)

oof_cb = np.zeros(len(X_raw), dtype=float)
oof_lgb = np.zeros(len(X_lgb), dtype=float)
test_cb = np.zeros(len(X_test_raw), dtype=float)
test_lgb = np.zeros(len(X_lgb_test), dtype=float)

n_seed_used = len({s for s, _, _ in splits})
t0 = time.time()
for i, (seed, tr_idx, va_idx) in enumerate(splits, 1):
    # CatBoost branch (raw/categorical)
    tr_pool = Pool(X_raw.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
    va_pool = Pool(X_raw.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
    te_pool = Pool(X_test_raw, cat_features=cat_cols)

    cb_model = CatBoostClassifier(**cb_params, random_seed=seed)
    cb_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

    oof_cb[va_idx] += cb_model.predict_proba(va_pool)[:, 1] / n_seed_used
    test_cb += cb_model.predict_proba(te_pool)[:, 1] / len(splits)

    # LGB branch (selected numeric matrix)
    X_tr = X_lgb.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X_lgb.iloc[va_idx]
    y_va = y.iloc[va_idx]

    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va)
    booster = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=6000 if not LIGHT_MODE else 3500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(stopping_rounds=250 if not LIGHT_MODE else 180, verbose=False)],
    )

    oof_lgb[va_idx] += booster.predict(X_va, num_iteration=booster.best_iteration) / n_seed_used
    test_lgb += booster.predict(X_lgb_test, num_iteration=booster.best_iteration) / len(splits)

    if i % 5 == 0 or i == len(splits):
        elapsed = (time.time() - t0) / 60
        log_step(f"[Train] fold {i}/{len(splits)} | elapsed={elapsed:.1f}m")

# =====================
# Tune 2-way ensemble (CB+LGB)
# =====================
min_pos = max(0.01, float(y.mean()) - OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None
max_pos = min(0.99, float(y.mean()) + OOF_POS_MARGIN) if USE_OOF_POS_CONSTRAINT else None

single_t_cb, single_f1_cb = find_best_threshold(y, oof_cb, n=981, min_pos_rate=min_pos, max_pos_rate=max_pos)
single_t_lgb, single_f1_lgb = find_best_threshold(y, oof_lgb, n=981, min_pos_rate=min_pos, max_pos_rate=max_pos)
print(f"[Single] CatBoost best_t={single_t_cb:.3f} | OOF F1={single_f1_cb:.4f}", flush=True)
print(f"[Single] LightGBM best_t={single_t_lgb:.3f} | OOF F1={single_f1_lgb:.4f}", flush=True)

best = tune_weights_threshold(y, oof_cb, oof_lgb, w_grid=W_CB_GRID)
log_step(f"Tuning finished: {best}")

test_ens = best["w_cb"] * test_cb + best["w_lgb"] * test_lgb

print("test_cb quantiles:", np.quantile(test_cb, [0, .01, .05, .1, .5, .9, .95, .99, 1]), flush=True)
print("test_lgb quantiles:", np.quantile(test_lgb, [0, .01, .05, .1, .5, .9, .95, .99, 1]), flush=True)
print("test_ens quantiles:", np.quantile(test_ens, [0, .01, .05, .1, .5, .9, .95, .99, 1]), flush=True)

# submission threshold
t_submit = float(best["t"]) if SUBMIT_USE_TUNED_THRESHOLD else threshold_for_target_pos_rate(test_ens, TARGET_POS_RATE_SINGLE)

if USE_SUBMISSION_POS_GUARD:
    train_pos_rate = float(y.mean())
    lower = max(0.01, train_pos_rate - OOF_POS_MARGIN)
    upper = min(0.99, train_pos_rate + OOF_POS_MARGIN)
    pos_rate_now = float((test_ens >= t_submit).mean())
    if (pos_rate_now < lower) or (pos_rate_now > upper):
        t_guard = threshold_for_target_pos_rate(test_ens, train_pos_rate)
        print(
            f"[SubmitGuard] pos_rate {pos_rate_now:.3f} out of [{lower:.3f}, {upper:.3f}] "
            f"-> threshold {t_submit:.5f} to {t_guard:.5f}",
            flush=True,
        )
        t_submit = float(t_guard)

test_pred = (test_ens >= t_submit).astype(int)
print("[Sanity] train pos_rate:", float(y.mean()), flush=True)
print("[Sanity] test pred pos_rate:", float(test_pred.mean()), flush=True)
print("[Sanity] t_submit:", float(t_submit), flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
SUB_PATH = "submission_3way_cb_lgb_tfidf.csv"
try:
    sub.to_csv(SUB_PATH, index=False)
    saved_path = SUB_PATH
except PermissionError:
    saved_path = "submission_3way_cb_lgb_tfidf_new.csv"
    sub.to_csv(saved_path, index=False)
    print(f"[Warn] {SUB_PATH} is locked; saved to {saved_path} instead.", flush=True)
print(
    f"Saved: {saved_path} | t_submit={t_submit:.5f} | mode={'tuned' if SUBMIT_USE_TUNED_THRESHOLD else 'target_pos'}",
    flush=True,
)
print(sub.head(), flush=True)

log_step("All done")
