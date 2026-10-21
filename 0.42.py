# ============================================================
# CatBoost + LightGBM Ensemble (실시간 현황 포함)
# - train.csv / test.csv만 사용
# - seed 3개 x 5fold 반복
# - CatBoost: categorical 직접 처리 (NaN -> "__MISSING__")
# - LightGBM: OOF Target Encoding + freq + label encoding
# - 앙상블 후 OOF threshold 튜닝 + 제출
# - 실시간: progress/ETA + 현재까지의 OOF F1@0.5 + pred pos_rate 출력
# ============================================================

import re, time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

# ----------------------------
# 0) Load
# ----------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "completed"
ID_COL = "ID"
y = train[TARGET].astype(int).copy()

# ----------------------------
# 1) Helpers
# ----------------------------
_SPLIT_RE = re.compile(r"\s*(?:,|/|;|\||、|\n|\t)\s*")

def split_tokens(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","null"}: return []
    toks = [t.strip() for t in _SPLIT_RE.split(s) if t.strip() != ""]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def add_multiselect_features(df, col, top_k=10, min_freq=20, prefix=None, add_binary=True, add_count=True):
    if prefix is None: prefix = col
    tokens_series = df[col].apply(split_tokens)

    if add_count:
        df[f"{prefix}__cnt"] = tokens_series.apply(len).astype(np.int16)

    if add_binary and top_k > 0:
        freq = {}
        for toks in tokens_series:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        cand = [(t,c) for t,c in freq.items() if c >= min_freq]
        cand.sort(key=lambda x: x[1], reverse=True)
        top_tokens = [t for t,_ in cand[:top_k]]

        for t in top_tokens:
            safe = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", t)[:40]
            df[f"{prefix}__has_{safe}"] = tokens_series.apply(lambda toks: int(t in toks)).astype(np.int8)
    return df

def add_text_summary(df, col, prefix=None):
    if prefix is None: prefix = col
    s = df[col].fillna("").astype(str)
    df[f"{prefix}__len"] = s.str.len().astype(np.int32)
    df[f"{prefix}__word_cnt"] = s.apply(lambda x: len(x.split())).astype(np.int16)
    df[f"{prefix}__has_digit"] = s.str.contains(r"\d", regex=True).astype(np.int8)
    df[f"{prefix}__has_alpha"] = s.str.contains(r"[A-Za-z]", regex=True).astype(np.int8)
    df[f"{prefix}__has_special"] = s.str.contains(r"[^0-9A-Za-z가-힣\s]", regex=True).astype(np.int8)
    df[f"{prefix}__is_empty"] = (s.str.strip() == "").astype(np.int8)
    return df

def find_best_threshold(y_true, y_prob, t_min=0.05, t_max=0.95, n=701):
    ts = np.linspace(t_min, t_max, n)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)

def fmt_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m{s:02d}s"

def progress_log(tag, step_no, total_steps, start_ts, step_durs, oof_partial=None, test_prob_partial=None):
    elapsed = time.time() - start_ts
    avg = float(np.mean(step_durs)) if step_durs else 0.0
    remain = total_steps - step_no
    eta = avg * remain
    msg = f"[{tag}] {step_no}/{total_steps} | elapsed={fmt_time(elapsed)} | avg={avg:.1f}s | ETA={fmt_time(eta)}"
    if oof_partial is not None:
        msg += f" | OOF F1@0.5(partial)={oof_partial:.4f}"
    if test_prob_partial is not None:
        msg += f" | test_pred_pos@0.5(partial)={test_prob_partial:.3f}"
    print(msg)

# ----------------------------
# 2) Column policy
# ----------------------------
DROP_COLS = ["idea_contest","contest_award","class4","contest_participation", ID_COL]
MS_SPLIT_AND_CNT = ["desired_certificate","major_field","desired_job"]
MS_CNT_ONLY = ["expected_domain","desired_job_except_data","onedayclass_topic","interested_company","previous_class_8"]
TEXT_SUMMARY_COLS = ["incumbents_lecture_scale_reason"]
CODELIKE_NUM_AS_CAT = ["generation","school1"]

# ----------------------------
# 3) Feature engineering
# ----------------------------
all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

for c in DROP_COLS:
    if c in all_df.columns: all_df.drop(columns=[c], inplace=True)

for c in CODELIKE_NUM_AS_CAT:
    if c in all_df.columns:
        all_df[c] = all_df[c].astype("Int64").astype(str).replace({"<NA>": np.nan})

for c in TEXT_SUMMARY_COLS:
    if c in all_df.columns:
        all_df = add_text_summary(all_df, c, prefix=c)
        all_df.drop(columns=[c], inplace=True)

for c in MS_SPLIT_AND_CNT:
    if c in all_df.columns:
        all_df = add_multiselect_features(all_df, c, top_k=10, min_freq=20, prefix=c, add_binary=True, add_count=True)
        all_df.drop(columns=[c], inplace=True)

for c in MS_CNT_ONLY:
    if c in all_df.columns:
        all_df = add_multiselect_features(all_df, c, top_k=0, min_freq=999999, prefix=c, add_binary=False, add_count=True)
        all_df.drop(columns=[c], inplace=True)

for c in all_df.columns:
    if all_df[c].dtype == "bool":
        all_df[c] = all_df[c].astype(np.int8)

X_all = all_df
X = X_all.iloc[:len(train)].copy()
X_test = X_all.iloc[len(train):].copy()

cat_cols = X.select_dtypes(include=["object","string","category"]).columns.tolist()

for c in cat_cols:
    X[c] = X[c].astype("string").fillna("__MISSING__")
    X_test[c] = X_test[c].astype("string").fillna("__MISSING__")

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
if num_cols:
    med = X[num_cols].median()
    X[num_cols] = X[num_cols].fillna(med)
    X_test[num_cols] = X_test[num_cols].fillna(med)

print(f"[Info] X={X.shape}, X_test={X_test.shape}, cat_cols={len(cat_cols)}, pos_rate={y.mean():.3f}")

# ----------------------------
# 4) CV splits
# ----------------------------
CV_SEEDS = [42, 202, 777]
N_SPLITS = 5
splits = []
for seed in CV_SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        splits.append((seed, tr_idx, va_idx))

total_steps = len(splits)

# ----------------------------
# 5) CatBoost OOF/Test (실시간 로그)
# ----------------------------
pos = int(y.sum()); neg = int(len(y)-pos)
pos_weight = neg / max(pos, 1)

cb_params = dict(
    iterations=4000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    class_weights=[1.0, float(pos_weight)],
    l2_leaf_reg=6,
    random_strength=1.0,
    od_type="Iter",
    od_wait=200,
)

oof_cb = np.zeros(len(X), dtype=float)
test_cb = np.zeros(len(X_test), dtype=float)

start_ts = time.time()
step_durs = []
for step_no, (seed, tr_idx, va_idx) in enumerate(splits, 1):
    t0 = time.time()

    tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
    va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
    te_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**cb_params, random_seed=seed)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

    oof_cb[va_idx] += model.predict_proba(va_pool)[:,1] / len(CV_SEEDS)
    test_cb += model.predict_proba(te_pool)[:,1] / total_steps

    dt = time.time() - t0
    step_durs.append(dt)

    if step_no % 3 == 0 or step_no == total_steps:
        oof_partial = f1_score(y, (oof_cb >= 0.5).astype(int))
        test_pos_partial = float((test_cb >= 0.5).mean())
        progress_log("CatBoost", step_no, total_steps, start_ts, step_durs, oof_partial, test_pos_partial)

# ----------------------------
# 6) LightGBM (OOF TE) 준비
# ----------------------------
def add_freq_encoding(train_df, test_df, cols):
    for c in cols:
        vc = train_df[c].value_counts(dropna=False)
        train_df[c + "__freq"] = train_df[c].map(vc).fillna(0).astype(np.float32)
        test_df[c + "__freq"] = test_df[c].map(vc).fillna(0).astype(np.float32)
    return train_df, test_df

def label_encode(train_df, test_df, cols):
    for c in cols:
        allv = pd.concat([train_df[c], test_df[c]], axis=0).astype("string")
        codes, _ = pd.factorize(allv, sort=False)
        train_df[c + "__le"] = codes[:len(train_df)].astype(np.int32)
        test_df[c + "__le"] = codes[len(train_df):].astype(np.int32)
    return train_df, test_df

def oof_target_encode_map(X_tr_col, y_tr, smooth=20):
    global_mean = float(y_tr.mean())
    stats = y_tr.groupby(X_tr_col).agg(["mean","count"])
    m = stats["mean"]; cnt = stats["count"]
    te_map = (m*cnt + global_mean*smooth) / (cnt + smooth)
    return te_map, global_mean

X_lgb = X.copy()
X_lgb_test = X_test.copy()

X_lgb, X_lgb_test = add_freq_encoding(X_lgb, X_lgb_test, cat_cols)
X_lgb, X_lgb_test = label_encode(X_lgb, X_lgb_test, cat_cols)

oof_lgb = np.zeros(len(X_lgb), dtype=float)
test_lgb = np.zeros(len(X_lgb_test), dtype=float)

lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
)

# ----------------------------
# 7) LightGBM OOF/Test (실시간 로그)
# ----------------------------
start_ts = time.time()
step_durs = []

for step_no, (seed, tr_idx, va_idx) in enumerate(splits, 1):
    t0 = time.time()

    X_tr = X_lgb.iloc[tr_idx].copy()
    y_tr = y.iloc[tr_idx]
    X_va = X_lgb.iloc[va_idx].copy()
    X_te = X_lgb_test.copy()

    # OOF Target Encoding: 원본 cat 컬럼 기반(누수 방지)
    for c in cat_cols:
        te_map, global_mean = oof_target_encode_map(X_tr[c], y_tr, smooth=20)
        X_tr[c + "__te"] = X_tr[c].map(te_map).fillna(global_mean).astype(np.float32)
        X_va[c + "__te"] = X_va[c].map(te_map).fillna(global_mean).astype(np.float32)
        X_te[c + "__te"] = X_te[c].map(te_map).fillna(global_mean).astype(np.float32)

    # raw cat 제거: le/freq/te만 사용
    X_tr2 = X_tr.drop(columns=cat_cols)
    X_va2 = X_va.drop(columns=cat_cols)
    X_te2 = X_te.drop(columns=cat_cols)

    dtr = lgb.Dataset(X_tr2, label=y_tr)
    dva = lgb.Dataset(X_va2, label=y.iloc[va_idx])

    booster = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=8000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )

    oof_lgb[va_idx] += booster.predict(X_va2, num_iteration=booster.best_iteration) / len(CV_SEEDS)
    test_lgb += booster.predict(X_te2, num_iteration=booster.best_iteration) / total_steps

    dt = time.time() - t0
    step_durs.append(dt)

    if step_no % 3 == 0 or step_no == total_steps:
        oof_partial = f1_score(y, (oof_lgb >= 0.5).astype(int))
        test_pos_partial = float((test_lgb >= 0.5).mean())
        progress_log("LightGBM", step_no, total_steps, start_ts, step_durs, oof_partial, test_pos_partial)

# ----------------------------
# 8) Ensemble + threshold tune
# ----------------------------
oof_ens = 0.5 * oof_cb + 0.5 * oof_lgb
test_ens = 0.5 * test_cb + 0.5 * test_lgb

t_best, f1_best = find_best_threshold(y, oof_ens, n=901)
print("\n====================")
print(f"[Ensemble OOF] Best threshold = {t_best:.3f} | Best F1 = {f1_best:.4f}")
print("====================\n")

test_pred = (test_ens >= t_best).astype(int)

print("[Sanity] train pos_rate:", float(y.mean()))
print("[Sanity] test pred pos_rate:", float(test_pred.mean()))

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
SUB_PATH = "submission_cb_lgb_ensemble_live.csv"
sub.to_csv(SUB_PATH, index=False)
print("Saved:", SUB_PATH)
print(sub.head())
