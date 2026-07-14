#!/usr/bin/env python
"""Parity instrument panel: quarreLM vs committed glmnet fixtures.

Run from anywhere inside the repo:  uv run tools/python/parity_check.py
"""

from pathlib import Path

import polars as pl

from quarrelm.api import enet

PARITY_TOL = 1e-6
SOLVER_TOL = 1e-12  # solver must be more converged than PARITY_TOL asserts
MAX_ITER = 100_000
VERBOSE = False  # print the offending coefficient rows per failing fit

DATASETS = {
    "diabetes": {
        "file": "benchmarks/data/diabetes.data.gz",
        "separator": "\t",
        "target": "Y",
        "drop": [],
        "drop_first_col": False,
    },
    "prostate": {
        "file": "benchmarks/data/prostate.data.gz",
        "separator": "\t",
        "target": "lpsa",
        "drop": ["train"],
        "drop_first_col": True,
    },
}


def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists():
            return p
    raise RuntimeError("not inside the quarreLM repo")


ROOT = repo_root()
FIXTURES = ROOT / "py-quarrelm" / "tests" / "fixtures"


def load_fixture(name: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURES / name, infer_schema_length=None)


def load_presets() -> pl.DataFrame:
    return load_fixture("parity_presets.csv").with_columns(
        pl.col("pf", "lower", "upper").map_elements(float, return_dtype=pl.Float64)
    )


def load_dataset(name: str) -> pl.DataFrame:
    cfg = DATASETS[name]
    df = pl.read_csv(
        ROOT / cfg["file"],
        separator=cfg["separator"],
        infer_schema_length=None,
    )
    if cfg["drop_first_col"]:
        df = df.drop(df.columns[0])
    df = df.drop([c for c in cfg["drop"] if c in df.columns])

    df = df.with_columns(pl.col(pl.String).str.strip_chars().cast(pl.Float64))

    return df


def check_dataset(
    name: str,
    df_data: pl.DataFrame,
    df_fits: pl.DataFrame,
    df_presets: pl.DataFrame,
) -> pl.DataFrame:
    """One quarreLM fit per fixture case; returns per-fit summary rows."""
    target = DATASETS[name]["target"]
    features = [c for c in df_data.columns if c != target]
    order = {f: i for i, f in enumerate(features)}

    summary = []
    partitions = df_fits.partition_by(
        ["preset", "alpha", "lambda"], as_dict=True, include_key=False
    )
    for (preset, alpha, lambda_), fix in partitions.items():
        ps = df_presets.filter(
            (pl.col("dataset") == name) & (pl.col("preset") == preset)
        ).sort(pl.col("feature").replace_strict(order, return_dtype=pl.Int64))
        assert ps.height == len(features), (
            f"{name}/{preset}: {ps.height} preset rows for {len(features)} features"
        )

        res = enet(
            df_data,
            target=target,
            alpha=alpha,
            lambda_=lambda_,
            penalty_factors=ps["pf"].to_numpy(),
            lower_bounds=ps["lower"].to_numpy(),
            upper_bounds=ps["upper"].to_numpy(),
            max_iter=MAX_ITER,
            tol=SOLVER_TOL,
        )
        ours = pl.DataFrame(
            {"feature": res.feature_names, "coef_quarrel": res.coef_array}
        )

        cmp = (
            fix.filter(pl.col("feature") != "(Intercept)")
            .join(ours, on="feature", how="left")
            .with_columns(
                (pl.col("coef") - pl.col("coef_quarrel")).abs().alias("abs_diff")
            )
        )

        summary.append(
            {
                "preset": preset,
                "alpha": alpha,
                "lambda": lambda_,
                "n_coef": cmp.height,
                "n_pass": cmp.filter(pl.col("abs_diff") <= PARITY_TOL).height,
                "n_unmatched": cmp.filter(pl.col("coef_quarrel").is_null()).height,
                "max_abs_diff": cmp["abs_diff"].max(),
            }
        )

        if VERBOSE:
            bad = cmp.filter(
                (pl.col("abs_diff") > PARITY_TOL) | pl.col("coef_quarrel").is_null()
            )
            if bad.height:
                print(f"\n-- {name} {preset} alpha={alpha} lambda={lambda_:.6g}")
                print(bad.select("feature", "coef", "coef_quarrel", "abs_diff"))

    return pl.DataFrame(summary).sort("max_abs_diff", descending=True, nulls_last=False)


def main() -> None:
    df_fits_all = load_fixture("parity_fits.csv")
    df_presets = load_presets()

    grand_pass = 0
    grand_coef = 0
    for name in DATASETS:
        fits = df_fits_all.filter(pl.col("dataset") == name)
        if fits.is_empty():
            print(f"\n=== {name}: no fixtures found, skipping ===")
            continue

        summary = check_dataset(name, load_dataset(name), fits, df_presets)
        n_pass, n_coef = summary["n_pass"].sum(), summary["n_coef"].sum()
        n_unmatched = summary["n_unmatched"].sum()
        grand_pass += n_pass
        grand_coef += n_coef

        print(f"\n=== {name}: {summary.height} fits ===")
        with pl.Config(tbl_rows=-1, float_precision=8):
            print(summary)
        print(
            f"  {n_pass}/{n_coef} coefs within {PARITY_TOL:g}"
            f"  (miss rate {1 - n_pass / n_coef:.4f}"
            f"{f', {n_unmatched} UNMATCHED FEATURES' if n_unmatched else ''})"
        )

    if grand_coef:
        print(
            f"\n=== overall: {grand_pass}/{grand_coef} "
            f"(miss rate {1 - grand_pass / grand_coef:.4f}) ==="
        )


if __name__ == "__main__":
    main()
