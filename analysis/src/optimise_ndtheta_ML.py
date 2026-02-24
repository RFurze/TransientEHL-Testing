#!/usr/bin/env python3
"""
Grid-search ND_FACTOR and standard ML regressors using existing 1:1 microscale results.

Workflow implemented in this script:
1) Build downsampled points from ``xi_rot.npy`` (same stage as ``prepare_microscale_tasks``).
2) Extract the true downsampled labels from full 1:1 results arrays.
3) Train a standard ML regressor on the downsampled points.
4) Predict at all non-downsampled points.
5) Measure error against the true 1:1 results and report the best ND/model settings.

Example:
python3 optimise_nd_ml.py \
  --output_dir OneToOne/ \
  --transient \
  --nd_factors "0.2,0.3,0.4,0.5,0.6" \
  --model ridge \
  --ridge_alphas "1e-6,1e-4,1e-2,1e0,1e2" \
  --match_tol 1e-12

Or:
python3 optimise_nd_ml.py \
  --output_dir OneToOne/ \
  --transient \
  --nd_factors "0.2,0.3,0.4,0.5,0.6" \
  --model knn \
  --knn_k "20,50,100,200" \
  --match_tol 1e-12

Notes:
- Evaluates on the held-out set (all points not selected by downsampling).
- Saves:
  - nd_ml_optimization.json
  - nd_ml_optimization.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

# Downsampling logic from your codebase
from coupling.src.functions.transient_coupling_classes import MetaModel3 as TransientMetaModel
from coupling.src.functions.coupling_classes import MetaModel3 as SteadyMetaModel

# Standard ML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


@dataclass
class VariableSpec:
    name: str
    values: np.ndarray
    feature_idx: list[int]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _as_feature_matrix(xi: np.ndarray, transient: bool) -> np.ndarray:
    # Matches your MLS script: [0,1,2,3,5,6,8,9] (+ [11,12] if transient)
    base_idx = [0, 1, 2, 3, 5, 6, 8, 9]
    if transient:
        base_idx += [11, 12]
    return np.vstack([xi[i] for i in base_idx]).T


def _as_feature_matrix_from_xid(xi_d: np.ndarray, transient: bool) -> np.ndarray:
    return _as_feature_matrix(xi_d, transient=transient)


def _match_downsample_indices(X_full: np.ndarray, X_down: np.ndarray, tol: float) -> np.ndarray:
    tree = cKDTree(X_full)
    dists, idx = tree.query(X_down, k=1)
    if np.any(dists > tol):
        raise ValueError(
            "Could not map some downsampled points back to xi_rot indices. "
            f"max_dist={dists.max():.3e}, tolerance={tol:.3e}."
        )
    print(f"Matching downsampled indices: Max distance = {dists.max():.3e}")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Downsampled points mapped to duplicate full indices; try tighter/more unique data.")
    return idx.astype(int)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    if denom < 1e-15:
        return _rmse(y_true, y_pred)
    return _rmse(y_true, y_pred) / denom


def _max_abs_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _max_rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-30)  # avoid div0 blow-ups
    return float(np.max(np.abs(y_true - y_pred) / denom))


def _iter_grid_1d(values: Iterable[float]):
    for v in values:
        yield v


def _iter_grid_2d(a: Iterable, b: Iterable):
    for x in a:
        for y in b:
            yield x, y


def _make_regressor(
    model: str,
    *,
    ridge_alpha: float = 1.0,
    knn_k: int = 50,
    knn_weights: str = "distance",
    rf_n_estimators: int = 200,
    rf_max_depth: Optional[int] = None,
    gbr_n_estimators: int = 200,
    gbr_learning_rate: float = 0.05,
    gbr_max_depth: int = 3,
    svr_c: float = 10.0,
    svr_gamma: str = "scale",
    svr_epsilon: float = 0.01,
    random_state: int = 0,
    n_jobs: int = 1,
) -> Pipeline:
    """
    Returns a sklearn Pipeline. We standardize features for models that benefit from it.
    """
    model = model.lower()

    if model == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=ridge_alpha, random_state=random_state)),
            ]
        )

    if model == "knn":
        # KNN strongly benefits from scaling when feature magnitudes differ.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", KNeighborsRegressor(n_neighbors=knn_k, weights=knn_weights)),
            ]
        )

    if model == "rf":
        # Scaling not needed.
        return Pipeline(
            steps=[
                ("reg", RandomForestRegressor(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ))
            ]
        )

    if model == "gbr":
        # Scaling not needed.
        return Pipeline(
            steps=[
                ("reg", GradientBoostingRegressor(
                    n_estimators=gbr_n_estimators,
                    learning_rate=gbr_learning_rate,
                    max_depth=gbr_max_depth,
                    random_state=random_state,
                ))
            ]
        )

    if model == "svr":
        # SVR needs scaling.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", SVR(C=svr_c, gamma=svr_gamma, epsilon=svr_epsilon)),
            ]
        )

    raise ValueError(f"Unknown --model '{model}'. Choose from: ridge, knn, rf, gbr, svr.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ND_FACTOR + ML regressor hyperparams on known 1:1 results")
    parser.add_argument("--output_dir", type=str, default="OneToOne/")
    parser.add_argument("--transient", action="store_true", help="Use transient 13-component xi layout")
    parser.add_argument("--nd_factors", type=str, required=True, help="Comma-separated ND_FACTOR values")
    parser.add_argument("--match_tol", type=float, default=1e-12)
    parser.add_argument("--prefer_high_nd_within", type=float, default=0.02,
                        help="Prefer higher ND if mean NRMSE is within this relative margin of the global minimum")

    # Model selection
    parser.add_argument("--model", type=str, required=True, help="ridge | knn | rf | gbr | svr")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)

    # Ridge grid
    parser.add_argument("--ridge_alphas", type=str, default="1.0", help="Comma-separated ridge alpha values")

    # KNN grid
    parser.add_argument("--knn_k", type=str, default="50", help="Comma-separated k values for KNN")
    parser.add_argument("--knn_weights", type=str, default="distance", choices=["uniform", "distance"])

    # RF grid
    parser.add_argument("--rf_n_estimators", type=str, default="200", help="Comma-separated n_estimators")
    parser.add_argument("--rf_max_depth", type=str, default="", help="Comma-separated max_depth (empty = None). Example: '5,10,20'")

    # GBR grid
    parser.add_argument("--gbr_n_estimators", type=str, default="200", help="Comma-separated n_estimators")
    parser.add_argument("--gbr_learning_rates", type=str, default="0.05", help="Comma-separated learning rates")
    parser.add_argument("--gbr_max_depths", type=str, default="3", help="Comma-separated max_depth values")

    # SVR grid
    parser.add_argument("--svr_c", type=str, default="10.0", help="Comma-separated C values")
    parser.add_argument("--svr_gamma", type=str, default="scale", help="Comma-separated gamma values (e.g. 'scale,auto,0.1')")
    parser.add_argument("--svr_epsilon", type=str, default="0.01", help="Comma-separated epsilon values")

    args = parser.parse_args()

    odir = args.output_dir
    nd_factors = _parse_csv_floats(args.nd_factors)
    print(f"nd_factors = {nd_factors}")
    print(f"model = {args.model}")

    xi_rot = np.load(os.path.join(odir, "xi_rot.npy"))
    X_full = _as_feature_matrix(xi_rot, transient=args.transient)
    n_full = X_full.shape[0]
    print(f"n_full: {n_full}")

    dq = np.load(os.path.join(odir, "dq_results.npy"))
    dp = np.load(os.path.join(odir, "dp_results.npy"))
    taust = np.load(os.path.join(odir, "tau_results.npy"))
    pmax = np.load(os.path.join(odir, "pmax_results.npy"))
    pmin = np.load(os.path.join(odir, "pmin_results.npy"))
    hmax = np.load(os.path.join(odir, "hmax_results.npy"))
    hmin = np.load(os.path.join(odir, "hmin_results.npy"))

    variables = [
        VariableSpec("dQx", dq[:, 0], [0, 1, 2, 4, 6] + ([8, 9] if args.transient else [])),
        VariableSpec("dQy", dq[:, 1], [0, 1, 3, 5, 7] + ([8, 9] if args.transient else [])),
        VariableSpec("dP", dp,       [0, 1, 4, 5, 6, 7] + ([8, 9] if args.transient else [])),
        VariableSpec("taustx", taust[:, 0], list(range(X_full.shape[1]))),
        VariableSpec("tausty", taust[:, 1], list(range(X_full.shape[1]))),
        VariableSpec("pmax", pmax, list(range(X_full.shape[1]))),
        VariableSpec("pmin", pmin, list(range(X_full.shape[1]))),
        VariableSpec("hmax", hmax, list(range(X_full.shape[1]))),
        VariableSpec("hmin", hmin, list(range(X_full.shape[1]))),
    ]

    model_cls = TransientMetaModel if args.transient else SteadyMetaModel

    # Build hyperparameter grid based on chosen model
    model_name = args.model.lower()
    grid = []

    if model_name == "ridge":
        alphas = _parse_csv_floats(args.ridge_alphas)
        for a in alphas:
            grid.append({"ridge_alpha": a})

    elif model_name == "knn":
        ks = _parse_csv_ints(args.knn_k)
        for k in ks:
            grid.append({"knn_k": k, "knn_weights": args.knn_weights})

    elif model_name == "rf":
        nes = _parse_csv_ints(args.rf_n_estimators)
        if args.rf_max_depth.strip():
            depths = [int(x) for x in args.rf_max_depth.replace(" ", "").split(",") if x]
        else:
            depths = [None]
        for ne, d in _iter_grid_2d(nes, depths):
            grid.append({"rf_n_estimators": ne, "rf_max_depth": d})

    elif model_name == "gbr":
        nes = _parse_csv_ints(args.gbr_n_estimators)
        lrs = _parse_csv_floats(args.gbr_learning_rates)
        mds = _parse_csv_ints(args.gbr_max_depths)
        for ne in nes:
            for lr in lrs:
                for md in mds:
                    grid.append({"gbr_n_estimators": ne, "gbr_learning_rate": lr, "gbr_max_depth": md})

    elif model_name == "svr":
        cs = _parse_csv_floats(args.svr_c)
        eps = _parse_csv_floats(args.svr_epsilon)
        gammas = [x for x in args.svr_gamma.replace(" ", "").split(",") if x]
        for c in cs:
            for e in eps:
                for g in gammas:
                    # try float gamma if user provided numeric
                    try:
                        g_val = float(g)
                    except ValueError:
                        g_val = g
                    grid.append({"svr_c": c, "svr_epsilon": e, "svr_gamma": g_val})
    else:
        raise ValueError(f"Unknown model '{args.model}'.")

    print(f"Hyperparameter configs: {len(grid)}")

    records: list[dict] = []

    for nd in nd_factors:
        print(f"\n=== ND_FACTOR={nd} ===")
        metamodel = model_cls(Nd_factor=nd)
        _, xi_d = metamodel.build(xi_rot, order=None, init=True, theta=None)
        print(f"Shape of xi_d: {np.shape(xi_d)}")

        X_down = _as_feature_matrix_from_xid(xi_d, transient=args.transient)
        idx_down = _match_downsample_indices(X_full, X_down, tol=args.match_tol)

        mask_down = np.zeros(n_full, dtype=bool)
        mask_down[idx_down] = True
        idx_query = np.where(~mask_down)[0]

        if len(idx_query) == 0:
            raise ValueError("Downsampling selected all points; cannot evaluate prediction on held-out points.")

        downsample_ratio = float(len(idx_down) / n_full)

        for cfg in grid:
            reg = _make_regressor(
                model_name,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                **cfg,
            )

            per_var = {}
            nrmse_values = []
            max_abs_list = []
            max_rel_list = []

            for var in variables:
                X_train = X_down[:, var.feature_idx]
                X_test = X_full[idx_query][:, var.feature_idx]

                y_train = var.values[idx_down]
                y_true = var.values[idx_query]

                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                rmse = _rmse(y_true, y_pred)
                nrmse = _nrmse(y_true, y_pred)
                max_abs_err = _max_abs_err(y_true, y_pred)
                max_rel_err = _max_rel_err(y_true, y_pred)

                per_var[var.name] = {
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "max_abs_err": max_abs_err,
                    "max_rel_err": max_rel_err,
                }

                nrmse_values.append(nrmse)
                max_abs_list.append(max_abs_err)
                max_rel_list.append(max_rel_err)

            mean_nrmse = float(np.mean(nrmse_values))

            rec = {
                "nd_factor": nd,
                "model": model_name,
                "hyperparams": cfg,
                "n_downsampled": int(len(idx_down)),
                "n_query": int(len(idx_query)),
                "downsample_ratio": downsample_ratio,
                "mean_nrmse": mean_nrmse,
                "max_abs_err": float(np.max(max_abs_list)) if max_abs_list else float("nan"),
                "max_rel_err": float(np.max(max_rel_list)) if max_rel_list else float("nan"),
                "per_variable": per_var,
            }
            records.append(rec)

            # concise print
            hp = ",".join([f"{k}={v}" for k, v in cfg.items()]) if cfg else "-"
            print(f"  cfg: {hp:40s}  down={downsample_ratio:.3f}  mean_nrmse={mean_nrmse:.6e}")

    # Global best: lowest mean_nrmse; tie-break by higher ND_FACTOR.
    records_sorted = sorted(records, key=lambda r: (r["mean_nrmse"], -r["nd_factor"]))
    global_best = records_sorted[0]

    # Prefer higher ND among solutions close to min error.
    min_err = global_best["mean_nrmse"]
    cutoff = min_err * (1.0 + args.prefer_high_nd_within)
    near_best = [r for r in records if r["mean_nrmse"] <= cutoff]
    preferred_high_nd = sorted(near_best, key=lambda r: (-r["nd_factor"], r["mean_nrmse"]))[0]

    payload = {
        "global_best": global_best,
        "preferred_high_nd": preferred_high_nd,
        "all_results": records_sorted,
        "settings": {
            "transient": bool(args.transient),
            "match_tol": float(args.match_tol),
            "prefer_high_nd_within": float(args.prefer_high_nd_within),
            "model": model_name,
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
        },
    }

    out_json = os.path.join(odir, "nd_ml_optimization.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_csv = os.path.join(odir, "nd_ml_optimization.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("nd_factor,model,hyperparams,n_downsampled,n_query,downsample_ratio,mean_nrmse,max_abs_err,max_rel_err\n")
        for r in records_sorted:
            hp = json.dumps(r["hyperparams"], sort_keys=True)
            f.write(
                f"{r['nd_factor']},{r['model']},{hp},{r['n_downsampled']},{r['n_query']},"
                f"{r['downsample_ratio']},{r['mean_nrmse']},{r['max_abs_err']},{r['max_rel_err']}\n"
            )

    print("\n=== Optimization summary ===")
    print(
        "Global best (min error): "
        f"ND_FACTOR={global_best['nd_factor']}, model={global_best['model']}, hyperparams={global_best['hyperparams']}, "
        f"downsample_ratio={global_best['downsample_ratio']:.3f}, mean_nrmse={global_best['mean_nrmse']:.6e}"
    )
    print(
        f"Preferred high-downsampling within {args.prefer_high_nd_within*100:.1f}% error margin: "
        f"ND_FACTOR={preferred_high_nd['nd_factor']}, model={preferred_high_nd['model']}, hyperparams={preferred_high_nd['hyperparams']}, "
        f"downsample_ratio={preferred_high_nd['downsample_ratio']:.3f}, mean_nrmse={preferred_high_nd['mean_nrmse']:.6e}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()