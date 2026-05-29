"""
The config CSV must have columns ``ticker``, ``start``, ``stop`` (extra
columns are ignored).
example
    ticker,start,stop
    ^GSPC,2010-01-01,2020-01-01
    ^GSPC,2000-01-01,2010-01-01
    ^GSPC,1990-01-01,2000-01-01
    ^GSPC,1980-01-01,1990-01-01

Outputs (default directory ./out):

    out/summary.csv                          -- one wide row per ticker x period
    out/<ticker>_<y0>_<y1>_persistence.png
    out/<ticker>_<y0>_<y1>_acf.png
    out/<ticker>_<y0>_<y1>_density.png
    out/<ticker>_<y0>_<y1>_bootstrap.png

"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from nig_hmm import (
    emission_moments, emission_logdens, stationary,
    baum_welch, decode, sojourn, simulate, acf,
)


MODELS = {
    "Gaussian":  ["norm", "norm"],
    "Student-t": ["norm", "t"],
    "NIG":       ["nig",  "nig"],
}
# short keys used as CSV column prefixes
MODEL_KEY = {"Gaussian": "gauss", "Student-t": "t", "NIG": "nig"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def get_returns(ticker, start, stop):
    df = yf.download(ticker, start=start, end=stop, progress=False)
    if df is None or len(df) < 2:
        raise RuntimeError(
            f"no price data returned for {ticker!r} between {start} and {stop}")
    prices = np.asarray(df["Close"]).ravel()
    dates = df.index[1:]  # diff drops the first observation
    r = 100 * np.diff(np.log(prices))
    return r, dates


# ---------------------------------------------------------------------------
# per-ticker pipeline
# ---------------------------------------------------------------------------
def run_one(ticker, start, stop, out, bootstrap_B,
            sim_len_acf, sim_len_moments):
    y0 = pd.Timestamp(start).year
    y1 = pd.Timestamp(stop).year
    tag   = f"{ticker}_{y0}_{y1}"
    title = f"{ticker} {y0} to {y1}"

    print(f"\n=== {title} ===")
    x, dates = get_returns(ticker, start, stop)
    print(f"  T={len(x)}  mean={x.mean():.3f}  sd={x.std():.3f}  "
          f"skew={((x-x.mean())**3).mean()/x.std()**3:.3f}  "
          f"kurt={((x-x.mean())**4).mean()/x.std()**4:.2f}")

    row = {"ticker": ticker, "start": start, "stop": stop,
           "n_obs": len(x),
           "data_skew": ((x - x.mean()) ** 3).mean() / x.std() ** 3,
           "data_kurt": ((x - x.mean()) ** 4).mean() / x.std() ** 4,
           "status": "ok"}

    # ---- fit
    res = {name: baum_welch(x, 2, kinds, seed=0) for name, kinds in MODELS.items()}
    for n, r in res.items():
        print(f"  {n:10s}  iters={r['n_iter']:3d}  BIC={r['bic']:.1f}")
    bic_min = min(r["bic"] for r in res.values())

    # ---- per-model: BIC, params, moments, persistence, simulated moments
    for name, r in res.items():
        k = MODEL_KEY[name]
        row[f"{k}_logL"] = r["loglik"]
        row[f"{k}_npar"] = r["n_par"]
        row[f"{k}_BIC"]  = r["bic"]
        row[f"{k}_dBIC"] = r["bic"] - bic_min
        row[f"{k}_iter"] = r["n_iter"]

        for j in range(r["m"]):
            mean, sd, sk, ek = emission_moments(r["kinds"][j], r["params"][j])
            sj = j + 1
            row[f"{k}_s{sj}_dist"]   = r["kinds"][j]
            row[f"{k}_s{sj}_Pstay"]  = r["P"][j, j]
            row[f"{k}_s{sj}_mean"]   = mean
            row[f"{k}_s{sj}_sd"]     = sd
            row[f"{k}_s{sj}_skew"]   = sk
            row[f"{k}_s{sj}_exkurt"] = ek

        st, _ = decode(r, x)
        sjn = sojourn(st, 2)
        row[f"{k}_sojourn_lo"]  = sjn[0]
        row[f"{k}_sojourn_hi"]  = sjn[1]
        row[f"{k}_transitions"] = int((np.diff(st) != 0).sum())

        sim = simulate(r, sim_len_moments, seed=5)
        sim_c = sim - sim.mean()
        row[f"{k}_sim_skew"] = (sim_c ** 3).mean() / sim.std() ** 3
        row[f"{k}_sim_kurt"] = (sim_c ** 4).mean() / sim.std() ** 4

    # ---- raw NIG parameters per state
    for j in range(2):
        for p in ("mu", "delta", "alpha", "beta"):
            row[f"nig_s{j+1}_{p}"] = res["NIG"]["params"][j][p]

    # ---- NIG seed-stability: spread of BIC across 5 seeds
    bic_seeds = []
    for s in range(5):
        rs = baum_welch(x, 2, ["nig", "nig"], seed=s)
        bic_seeds.append(rs["bic"])
    row["nig_seed_bic_std"] = float(np.std(bic_seeds, ddof=1))

    # ---- bootstrap on the NIG fit
    base = res["NIG"]
    fitted_beta = base["params"][1]["beta"]
    beta_bs = np.empty(bootstrap_B)
    for b in range(bootstrap_B):
        xb = simulate(base, len(x), seed=1000 + b)
        rb = baum_welch(xb, 2, ["nig", "nig"], n_init=3, seed=b)
        beta_bs[b] = rb["params"][1]["beta"]
    row["nig_boot_B"]      = bootstrap_B
    row["nig_boot_beta"]   = fitted_beta
    row["nig_boot_mean"]   = float(beta_bs.mean())
    row["nig_boot_se"]     = float(beta_bs.std(ddof=1))
    row["nig_boot_ci_lo"]  = float(np.percentile(beta_bs, 2.5))
    row["nig_boot_ci_hi"]  = float(np.percentile(beta_bs, 97.5))

    print(f"  bootstrap beta = {fitted_beta:.3f}  "
          f"mean={beta_bs.mean():.3f}  SE={beta_bs.std(ddof=1):.3f}  "
          f"CI=[{np.percentile(beta_bs,2.5):.3f}, {np.percentile(beta_bs,97.5):.3f}]")

    # ---- plots
    _plot_persistence(res, x, dates, tag, title, out)
    _plot_acf(res, x, tag, title, out, sim_len_acf)
    _plot_density(res, x, tag, title, out)
    _plot_bootstrap(beta_bs, fitted_beta, tag, title, out)
    return row


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def _save(out, name):
    path = os.path.join(out, name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def _plot_persistence(res, x, dates, tag, title, out):
    fig, ax = plt.subplots(len(res), 1, figsize=(11, 6), sharex=True)
    for a, (n, r) in zip(ax, res.items()):
        st, _ = decode(r, x)
        a.plot(dates, x, lw=0.4, color="0.4")
        a.fill_between(dates, x.min(), x.max(), where=(st == 1),
                       color="crimson", alpha=0.2, step="mid")
        a.set_ylabel(n)
        a.set_xlim(dates[0], dates[-1])
    ax[0].set_title(f"{title}: returns with high-volatility state shaded")
    ax[-1].set_xlabel("date")
    fig.autofmt_xdate()
    plt.tight_layout()
    _save(out, f"{tag}_persistence.png")


def _plot_acf(res, x, tag, title, out, sim_len):
    lags = np.arange(1, 100)
    plt.figure(figsize=(11, 5))
    plt.bar(lags, acf(np.abs(x), lags), color="0.8", label="empirical")
    for n, r in res.items():
        sim = np.abs(simulate(r, sim_len, seed=2))
        plt.plot(lags, acf(sim, lags), lw=1.6, label=n)
    plt.xlabel("lag"); plt.ylabel("ACF of |returns|")
    plt.title(f"{title}: autocorrelation of absolute returns")
    plt.legend()
    _save(out, f"{tag}_acf.png")


def _plot_density(res, x, tag, title, out):
    def model_pdf(r, grid):
        lB = emission_logdens(grid, r["kinds"], r["params"])
        return (np.exp(lB) * stationary(r["P"])).sum(1)
    grid = np.linspace(x.min(), x.max(), 600)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, logy in zip(ax, [False, True]):
        a.hist(x, bins=120, density=True, color="0.85", label="data")
        for n, r in res.items():
            a.plot(grid, model_pdf(r, grid), lw=1.6, label=n)
        a.set_yscale("log" if logy else "linear")
        a.set_title("tails (log scale)" if logy else "centre (linear scale)")
        a.set_xlabel("return [%]"); a.legend()
    ax[0].set_ylabel("density")
    plt.suptitle(title)
    plt.tight_layout()
    _save(out, f"{tag}_density.png")


def _plot_bootstrap(beta_bs, fitted_beta, tag, title, out):
    plt.figure(figsize=(8, 3.5))
    plt.hist(beta_bs, bins=max(10, len(beta_bs) // 3), color="0.7", edgecolor="w")
    plt.axvline(fitted_beta, color="crimson", lw=2, label="fitted")
    plt.axvline(0, color="k", ls="--", lw=1, label="no skew")
    plt.xlabel("bootstrap beta (volatile state)")
    plt.title(f"{title}: bootstrap distribution of volatile-state beta")
    plt.legend()
    _save(out, f"{tag}_bootstrap.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Batch HMM experiment runner driven by a (ticker, start, stop) CSV.")
    ap.add_argument("config", help="CSV with columns: ticker, start, stop")
    ap.add_argument("--out", default="out",
                    help="output directory for plots and summary.csv (default ./out)")
    ap.add_argument("--summary", default="summary.csv",
                    help="filename for the consolidated CSV (default summary.csv)")
    ap.add_argument("--bootstrap-B", type=int, default=50,
                    help="number of parametric-bootstrap refits per ticker (default 12)")
    ap.add_argument("--sim-len-acf", type=int, default=400000,
                    help="simulated series length for ACF plot (default 400000)")
    ap.add_argument("--sim-len-moments", type=int, default=800000,
                    help="simulated series length for skew/kurtosis (default 800000)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)

    cfg = pd.read_csv(args.config)
    cfg.columns = [c.strip().lower() for c in cfg.columns]
    missing = {"ticker", "start", "stop"} - set(cfg.columns)
    if missing:
        sys.exit(f"config CSV missing required columns: {sorted(missing)}")

    rows = []
    for _, cr in cfg.iterrows():
        ticker = str(cr["ticker"]).strip()
        start, stop = str(cr["start"]).strip(), str(cr["stop"]).strip()
        try:
            rows.append(run_one(
                ticker, start, stop, args.out,
                bootstrap_B=args.bootstrap_B,
                sim_len_acf=args.sim_len_acf,
                sim_len_moments=args.sim_len_moments,
            ))
        except Exception as e:
            print(f"  !! failed: {e}", file=sys.stderr)
            traceback.print_exc()
            rows.append({"ticker": ticker, "start": start, "stop": stop,
                         "status": f"error: {e}"})

    summary_path = os.path.join(args.out, args.summary)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n=== done. {len(rows)} tickers processed; summary -> {summary_path} ===")


if __name__ == "__main__":
    main()