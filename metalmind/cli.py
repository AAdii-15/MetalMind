"""Command-line interface.

    python -m metalmind collect-news                     # run daily to build the news archive
    python -m metalmind evaluate --metal copper          # walk-forward backtest
    python -m metalmind forecast --metal copper          # 10-day forecast + chart
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from metalmind import __version__, evaluate, news, plotting, prices
from metalmind.config import DATA_DIR, METALS, OUTPUT_DIR, get_metal
from metalmind.forecast import forecast

log = logging.getLogger("metalmind")
LOW_COVERAGE = 0.2


def _load(args) -> tuple[pd.Series, str, pd.DataFrame | None]:
    series, source = prices.get_price_history(args.metal, args.prices_csv, Path(args.data_dir))
    print(f"Prices: {len(series)} days from {series.index[0]:%d %b %Y} to {series.index[-1]:%d %b %Y} ({source})")
    print(f"Latest {get_metal(args.metal).name} price: ₹{series.iloc[-1]:,.2f} per kg on {series.index[-1]:%d %b %Y}")
    if args.no_news:
        return series, source, None
    archive, new = news.collect(args.metal, Path(args.data_dir))
    sentiment = news.daily_sentiment(archive, series.index)
    covered = int((sentiment["news_count"] > 0).sum())
    print(f"News: {len(archive)} relevant articles archived ({new} new); {covered} of {len(series)} trading days have news")
    return series, source, sentiment


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:,.2f}")


def cmd_collect(args) -> int:
    for metal in args.metal or list(METALS):
        archive, new = news.collect(metal, Path(args.data_dir))
        print(f"{metal:<10} {new:>4} new articles, {len(archive):>5} in archive")
    return 0


def cmd_evaluate(args) -> int:
    series, _, sentiment = _load(args)
    bt = evaluate.run(series, sentiment, min_train=args.min_train, refit_every=args.refit_every)
    print(f"\nWalk-forward backtest: {len(bt.predictions)} test days "
          f"({bt.predictions.index[0]:%d %b %Y} to {bt.predictions.index[-1]:%d %b %Y}), next-day horizon\n")
    print(_fmt(bt.summary))
    if bt.news_coverage is not None and bt.news_coverage < LOW_COVERAGE:
        print(f"\nNote: only {bt.news_coverage:.0%} of test days have news, so the price-only vs price + news "
              "comparison is not meaningful yet. Run `collect-news` daily to build the archive.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.metal}_backtest_{date.today():%Y%m%d}"
    bt.predictions.to_csv(out / f"{stem}.csv")
    bt.summary.to_csv(out / f"{stem}_summary.csv")
    best = bt.summary.drop(index=["Naive (no change)", "Drift (mean return)"])["MAE (₹/kg)"].idxmin()
    png = plotting.plot_backtest(bt.predictions, best, args.metal, out / f"{stem}.png")
    print(f"\nSaved: {out / (stem + '.csv')}, {out / (stem + '_summary.csv')}, {png}")
    return 0


def cmd_forecast(args) -> int:
    series, _, sentiment = _load(args)
    resid = None
    try:
        bt = evaluate.run(series, sentiment, min_train=args.min_train, refit_every=args.refit_every)
        model = "Random Forest: price + news" if sentiment is not None else "Random Forest: price only"
        p = bt.predictions
        resid = float(np.std((p[model] - p["actual"]) / p["last_price"]))
        row = bt.summary.loc[model]
        print(f"\nBacktest of this model ({len(p)} days): MAE ₹{row['MAE (₹/kg)']:,.2f}/kg, "
              f"{row['MAE vs naive (%)']:+.1f}% vs the naive no-change baseline")
    except ValueError as exc:
        print(f"\nNo backtest (error band unavailable): {exc}")

    fc = forecast(series, sentiment, horizon=args.horizon, residual_std=resid)
    print(f"\n{args.horizon}-day forecast (₹ per kg):\n")
    print(_fmt(fc.rename_axis("date").rename(columns=str.title)))
    change = fc["forecast"].iloc[-1] / series.iloc[-1] - 1
    print(f"\nChange over horizon: {change:+.2%}. This is a research estimate, not a guaranteed price.")

    out = Path(args.output_dir)
    stem = f"{args.metal}_forecast_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out / f"{stem}.csv")
    if not args.no_plot:
        png = plotting.plot_forecast(series, fc, args.metal, out / f"{stem}.png")
        print(f"Saved: {out / (stem + '.csv')}, {png}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="metalmind", description="News-sentiment-aware metal price forecasting")
    ap.add_argument("--version", action="version", version=f"metalmind {__version__}")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="where price and news archives are kept")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR), help="where CSVs and charts are written")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect-news", help="fetch today's headlines into the news archive")
    c.add_argument("--metal", nargs="*", choices=list(METALS), help="default: all metals")
    c.set_defaults(func=cmd_collect)

    for name, func, helptext in (
        ("evaluate", cmd_evaluate, "walk-forward backtest against baselines"),
        ("forecast", cmd_forecast, "forecast the next days and save a chart"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--metal", required=True, choices=list(METALS))
        p.add_argument("--prices-csv", help="CSV with date and price columns (otherwise scrape BankBazaar)")
        p.add_argument("--no-news", action="store_true", help="price features only; skip news collection")
        p.add_argument("--min-train", type=int, default=60, help="days in the first training window")
        p.add_argument("--refit-every", type=int, default=5, help="refit models every N test days")
        if name == "forecast":
            p.add_argument("--horizon", type=int, default=10)
            p.add_argument("--no-plot", action="store_true")
        p.set_defaults(func=func)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        return args.func(args)
    except (prices.PriceDataError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
