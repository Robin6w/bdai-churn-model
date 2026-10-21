import copy
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET = "completed"
ID_COL = "ID"
BASELINE_SUB_PATH = "0218_0.433.csv"
TARGET_TEST_POS_RATE = 0.68059
CB_WEIGHT_FACTOR_SINGLE = 0.87

_SPLIT_RE = re.compile(r"\s*(?:,|;|\||、|\n|\t)\s*")


def split_tokens(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    s = s.replace(" / ", ";")
    toks = [t.strip() for t in _SPLIT_RE.split(s) if t.strip() != ""]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def add_multiselect_features(df, col, top_k=10, min_freq=20, prefix=None, add_binary=True, add_count=True):
    if prefix is None:
        prefix = col
    tokens_series = df[col].apply(split_tokens)

    if add_count:
        df[f"{prefix}__cnt"] = tokens_series.apply(len).astype(np.int16)

    if add_binary and top_k > 0:
        freq = {}
        for toks in tokens_series:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        cand = [(t, c) for t, c in freq.items() if c >= min_freq]
        cand.sort(key=lambda x: x[1], reverse=True)
        top_tokens = [t for t, _ in cand[:top_k]]

        for t in top_tokens:
            safe = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", t)[:40]
            df[f"{prefix}__has_{safe}"] = tokens_series.apply(lambda toks: int(t in toks)).astype(np.int8)
    return df


def add_text_summary(df, col, prefix=None):
    if prefix is None:
        prefix = col
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


def tune_weights_threshold(y_true, oof_cb, oof_lgb, w_grid, t_min=0.05, t_max=0.95, n_t=901):
    best = {"w_cb": 0.5, "w_lgb": 0.5, "t": 0.5, "f1": -1.0}
    for w_cb in w_grid:
        mix = w_cb * oof_cb + (1.0 - w_cb) * oof_lgb
        t, f1 = find_best_threshold(y_true, mix, t_min=t_min, t_max=t_max, n=n_t)
        if f1 > best["f1"]:
            best = {"w_cb": float(w_cb), "w_lgb": float(1.0 - w_cb), "t": float(t), "f1": float(f1)}
    return best


def threshold_for_target_pos_rate(y_prob, target_pos_rate):
    target_pos_rate = float(np.clip(target_pos_rate, 0.001, 0.999))
    q = float(np.clip(1.0 - target_pos_rate, 0.0, 1.0))
    return float(np.quantile(y_prob, q))


def fmt_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m{s:02d}s"


def build_features(train, test, cfg):
    drop_cols = ["idea_contest", "contest_award", "class4", "contest_participation", ID_COL]
    ms_split_and_cnt = ["desired_certificate", "major_field", "desired_job"]
    ms_cnt_only = ["expected_domain", "desired_job_except_data", "onedayclass_topic", "interested_company", "previous_class_8"]
    text_summary_cols = ["incumbents_lecture_scale_reason"]
    codelike_num_as_cat = ["generation", "school1"]

    all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

    for c in drop_cols:
        if c in all_df.columns:
            all_df.drop(columns=[c], inplace=True)

    for c in codelike_num_as_cat:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype("Int64").astype(str).replace({"<NA>": np.nan})

    for c in text_summary_cols:
        if c in all_df.columns:
            all_df = add_text_summary(all_df, c, prefix=c)
            all_df.drop(columns=[c], inplace=True)

    for c in ms_split_and_cnt:
        if c in all_df.columns:
            all_df = add_multiselect_features(
                all_df,
                c,
                top_k=cfg["ms_top_k"],
                min_freq=cfg["ms_min_freq"],
                prefix=c,
                add_binary=True,
                add_count=True,
            )
            all_df.drop(columns=[c], inplace=True)

    for c in ms_cnt_only:
        if c in all_df.columns:
            all_df = add_multiselect_features(
                all_df,
                c,
                top_k=0,
                min_freq=999999,
                prefix=c,
                add_binary=False,
                add_count=True,
            )
            all_df.drop(columns=[c], inplace=True)

    for c in all_df.columns:
        if all_df[c].dtype == "bool":
            all_df[c] = all_df[c].astype(np.int8)

    X = all_df.iloc[: len(train)].copy()
    X_test = all_df.iloc[len(train) :].copy()

    if cfg["add_missing_meta"]:
        X["row_missing_cnt"] = X.isna().sum(axis=1).astype(np.int16)
        X_test["row_missing_cnt"] = X_test.isna().sum(axis=1).astype(np.int16)
        X["row_missing_ratio"] = X.isna().mean(axis=1).astype(np.float32)
        X_test["row_missing_ratio"] = X_test.isna().mean(axis=1).astype(np.float32)

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__")

    if cfg["use_rare_bucket"]:
        for c in cat_cols:
            vc = X[c].value_counts(dropna=False)
            keep = set(vc[vc >= cfg["rare_min_freq"]].index)
            X[c] = X[c].where(X[c].isin(keep), "__RARE__")
            X_test[c] = X_test[c].where(X_test[c].isin(keep), "__RARE__")

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        med = X[num_cols].median()
        X[num_cols] = X[num_cols].fillna(med)
        X_test[num_cols] = X_test[num_cols].fillna(med)

    return X, X_test, cat_cols


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
        train_df[c + "__le"] = codes[: len(train_df)].astype(np.int32)
        test_df[c + "__le"] = codes[len(train_df) :].astype(np.int32)
    return train_df, test_df


def oof_target_encode_map(x_tr_col, y_tr, smooth=20):
    global_mean = float(y_tr.mean())
    stats = y_tr.groupby(x_tr_col).agg(["mean", "count"])
    m = stats["mean"]
    cnt = stats["count"]
    te_map = (m * cnt + global_mean * smooth) / (cnt + smooth)
    return te_map, global_mean


def build_splits(X, y, seeds, n_splits):
    splits = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X, y):
            splits.append((seed, tr_idx, va_idx))
    return splits


def run_experiment(train, test, y, cfg, baseline_pred=None):
    t_run = time.time()
    print(f"\n[Run] {cfg['name']} started")

    X, X_test, cat_cols = build_features(train, test, cfg)
    print(f"[Run] {cfg['name']} features: X={X.shape}, X_test={X_test.shape}, cat={len(cat_cols)}")

    splits = build_splits(X, y, cfg["cv_seeds"], cfg["n_splits"])
    total_steps = len(splits)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_weight = neg / max(pos, 1)

    cb_params = dict(
        iterations=cfg["cb_iterations"],
        learning_rate=cfg["cb_learning_rate"],
        depth=cfg["cb_depth"],
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        l2_leaf_reg=cfg["cb_l2_leaf_reg"],
        random_strength=cfg["cb_random_strength"],
        od_type="Iter",
        od_wait=cfg["cb_od_wait"],
    )
    if cfg["use_cb_class_weight"]:
        cb_params["class_weights"] = [1.0, float(pos_weight * cfg["cb_weight_factor"])]

    oof_cb = np.zeros(len(X), dtype=float)
    test_cb = np.zeros(len(X_test), dtype=float)

    t0 = time.time()
    for step_no, (seed, tr_idx, va_idx) in enumerate(splits, 1):
        tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        te_pool = Pool(X_test, cat_features=cat_cols)

        model = CatBoostClassifier(**cb_params, random_seed=seed)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        oof_cb[va_idx] += model.predict_proba(va_pool)[:, 1] / len(cfg["cv_seeds"])
        test_cb += model.predict_proba(te_pool)[:, 1] / total_steps

        if step_no % max(1, total_steps // 3) == 0 or step_no == total_steps:
            print(f"[Run] {cfg['name']} CatBoost {step_no}/{total_steps} elapsed={fmt_time(time.time()-t0)}")

    X_lgb = X.copy()
    X_lgb_test = X_test.copy()
    X_lgb, X_lgb_test = add_freq_encoding(X_lgb, X_lgb_test, cat_cols)
    X_lgb, X_lgb_test = label_encode(X_lgb, X_lgb_test, cat_cols)

    oof_lgb = np.zeros(len(X_lgb), dtype=float)
    test_lgb = np.zeros(len(X_lgb_test), dtype=float)

    lgb_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=cfg["lgb_learning_rate"],
        num_leaves=cfg["lgb_num_leaves"],
        min_data_in_leaf=cfg["lgb_min_data_in_leaf"],
        feature_fraction=cfg["lgb_feature_fraction"],
        bagging_fraction=cfg["lgb_bagging_fraction"],
        bagging_freq=1,
        lambda_l2=cfg["lgb_lambda_l2"],
        verbose=-1,
    )

    t1 = time.time()
    for step_no, (_, tr_idx, va_idx) in enumerate(splits, 1):
        X_tr = X_lgb.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx]
        X_va = X_lgb.iloc[va_idx].copy()
        X_te = X_lgb_test.copy()

        for c in cat_cols:
            te_map, global_mean = oof_target_encode_map(X_tr[c], y_tr, smooth=cfg["te_smooth"])
            X_tr[c + "__te"] = X_tr[c].map(te_map).fillna(global_mean).astype(np.float32)
            X_va[c + "__te"] = X_va[c].map(te_map).fillna(global_mean).astype(np.float32)
            X_te[c + "__te"] = X_te[c].map(te_map).fillna(global_mean).astype(np.float32)

        X_tr2 = X_tr.drop(columns=cat_cols)
        X_va2 = X_va.drop(columns=cat_cols)
        X_te2 = X_te.drop(columns=cat_cols)

        dtr = lgb.Dataset(X_tr2, label=y_tr)
        dva = lgb.Dataset(X_va2, label=y.iloc[va_idx])

        booster = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=cfg["lgb_num_boost_round"],
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=cfg["lgb_early_stopping"], verbose=False)],
        )

        oof_lgb[va_idx] += booster.predict(X_va2, num_iteration=booster.best_iteration) / len(cfg["cv_seeds"])
        test_lgb += booster.predict(X_te2, num_iteration=booster.best_iteration) / total_steps

        if step_no % max(1, total_steps // 3) == 0 or step_no == total_steps:
            print(f"[Run] {cfg['name']} LightGBM {step_no}/{total_steps} elapsed={fmt_time(time.time()-t1)}")

    t_cb, f1_cb = find_best_threshold(y, oof_cb, n=cfg["threshold_points"])
    t_lgb, f1_lgb = find_best_threshold(y, oof_lgb, n=cfg["threshold_points"])
    best_ens = tune_weights_threshold(
        y,
        oof_cb,
        oof_lgb,
        cfg["w_grid"],
        n_t=cfg["threshold_points"],
    )

    test_ens = best_ens["w_cb"] * test_cb + best_ens["w_lgb"] * test_lgb
    t_submit = threshold_for_target_pos_rate(test_ens, TARGET_TEST_POS_RATE)
    pred = (test_ens >= t_submit).astype(int)
    test_pos_rate = float(pred.mean())

    diff_count = None
    if baseline_pred is not None and len(baseline_pred) == len(pred):
        diff_count = int((baseline_pred != pred).sum())

    robust_score = best_ens["f1"]
    if diff_count is not None and diff_count > 160:
        robust_score -= 0.00012 * (diff_count - 160)

    elapsed = time.time() - t_run
    print(
        f"[Run] {cfg['name']} done | OOF={best_ens['f1']:.4f} | "
        f"w_cb={best_ens['w_cb']:.2f} | t_submit={t_submit:.5f} | "
        f"test_pos={test_pos_rate:.5f} | diff={diff_count} | elapsed={fmt_time(elapsed)}"
    )

    return {
        "name": cfg["name"],
        "oof_f1_cb": float(f1_cb),
        "oof_f1_lgb": float(f1_lgb),
        "oof_f1_ens": float(best_ens["f1"]),
        "w_cb": float(best_ens["w_cb"]),
        "w_lgb": float(best_ens["w_lgb"]),
        "t_cb": float(t_cb),
        "t_lgb": float(t_lgb),
        "t_ens": float(best_ens["t"]),
        "t_submit": float(t_submit),
        "test_pos_rate": test_pos_rate,
        "diff_count": diff_count,
        "robust_score": float(robust_score),
        "pred": pred,
    }


def get_base_config():
    return {
        "name": "baseline",
        "cv_seeds": [42, 202],
        "n_splits": 5,
        "ms_top_k": 10,
        "ms_min_freq": 20,
        "add_missing_meta": False,
        "use_rare_bucket": False,
        "rare_min_freq": 5,
        "cb_iterations": 3000,
        "cb_learning_rate": 0.05,
        "cb_depth": 6,
        "cb_l2_leaf_reg": 6,
        "cb_random_strength": 1.0,
        "cb_od_wait": 200,
        "use_cb_class_weight": True,
        "cb_weight_factor": 1.0,
        "lgb_learning_rate": 0.03,
        "lgb_num_leaves": 64,
        "lgb_min_data_in_leaf": 20,
        "lgb_feature_fraction": 0.8,
        "lgb_bagging_fraction": 0.8,
        "lgb_lambda_l2": 1.0,
        "lgb_num_boost_round": 6000,
        "lgb_early_stopping": 300,
        "te_smooth": 20,
        "w_grid": np.linspace(0.0, 1.0, 51),
        "threshold_points": 701,
    }


def apply_overrides(cfg, name, overrides):
    out = copy.deepcopy(cfg)
    out["name"] = name
    for k, v in overrides.items():
        out[k] = v
    return out


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].astype(int).copy()

    baseline_pred = None
    if pd.io.common.file_exists(BASELINE_SUB_PATH):
        baseline_df = pd.read_csv(BASELINE_SUB_PATH).sort_values(ID_COL).reset_index(drop=True)
        if TARGET in baseline_df.columns and len(baseline_df) == len(test):
            baseline_pred = baseline_df[TARGET].astype(int).to_numpy()
            print(f"[Info] baseline loaded: {BASELINE_SUB_PATH}")

    cfg = get_base_config()
    cfg["name"] = "cb_weight_0p75_single"
    cfg["cv_seeds"] = [42, 202, 777]
    cfg["cb_iterations"] = 4000
    cfg["lgb_num_boost_round"] = 8000
    cfg["w_grid"] = np.linspace(0.0, 1.0, 101)
    cfg["threshold_points"] = 901
    cfg["cb_weight_factor"] = CB_WEIGHT_FACTOR_SINGLE
    cfg["use_cb_class_weight"] = True

    res = run_experiment(train, test, y, cfg, baseline_pred=baseline_pred)

    summary = pd.DataFrame(
        [
            {
                "name": res["name"],
                "oof_f1_cb": res["oof_f1_cb"],
                "oof_f1_lgb": res["oof_f1_lgb"],
                "oof_f1_ens": res["oof_f1_ens"],
                "w_cb": res["w_cb"],
                "w_lgb": res["w_lgb"],
                "t_submit": res["t_submit"],
                "test_pos_rate": res["test_pos_rate"],
                "diff_count": res["diff_count"],
                "robust_score": res["robust_score"],
                "cb_weight_factor": CB_WEIGHT_FACTOR_SINGLE,
            }
        ]
    )
    factor_tag = str(CB_WEIGHT_FACTOR_SINGLE).replace(".", "p")
    summary_path = f"0222_single_summary_cbw{factor_tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: res["pred"].astype(int)})
    out_path = f"submission_0222_cbw{factor_tag}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(
        f"[Final] oof_f1={res['oof_f1_ens']:.4f} | w_cb={res['w_cb']:.2f} | "
        f"t_submit={res['t_submit']:.5f} | diff={res['diff_count']}"
    )


if __name__ == "__main__":
    main()
