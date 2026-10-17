# MetalMind

**News-sentiment-aware price forecasting for industrial metals in India (₹/kg).**

MetalMind collects daily metal prices and metal-related business news, scores the news with sentiment analysis, and forecasts prices for the coming days. Every model is backtested against simple baselines, so you can see whether it actually adds value before trusting a forecast.

Supported metals: copper, aluminium, steel, lead, zinc, nickel.

## Why

Small and medium manufacturers buy steel, zinc, aluminium and copper in bulk, often after quoting a price to their own customers. MetalMind started during my internship at **Shri Ram Products, Jaipur** (a manufacturer of power-line hardware). The question was whether public news and recent price behaviour can give early warning of raw-material price moves.

## How it works

```
 prices.py         news.py                features.py          evaluate.py           forecast.py
 ─────────         ───────                ───────────          ───────────           ───────────
 CSV or       ┐    RSS feeds  → keyword   lagged returns,      walk-forward backtest  recursive
 BankBazaar   ├──► filter → de-dup →  ──► volatility,     ──►  vs naive and drift ──► 10-day forecast
 (archived)   ┘    VADER sentiment        momentum, daily      baselines (MAE, RMSE,  with error band
                   (archived daily)       sentiment, news      MAPE, direction)       + chart
                                          volume
```

1. **Prices.** Load a daily price history from a CSV (recommended: an MCX/LME export or your purchase records), or scrape BankBazaar's recent price table. Scraped prices are merged into a local archive so the history grows with each run. If no real data is available the tool stops with a clear error; it never substitutes made-up prices.
2. **News.** Headlines from Livemint, Economic Times and a metal-specific Google News search are filtered with word-boundary keyword rules, so "lead" does not match "leadership" and "iron" does not match "environment". Duplicates and syndicated copies are removed, and each article is scored with VADER. RSS feeds only show their latest items, so `collect-news` appends to an archive. Run it daily to build up history.
3. **Alignment.** Each article is assigned to the first trading day on or after its publication date (IST), so weekend news counts towards Monday. Features for day *t* use only information available by day *t*. The target is the return from *t* to *t+1*.
4. **Models.** Models predict the **forward return**, not the price level, because tree models cannot extrapolate outside the range of target values seen in training. Ridge regression, Random Forest and Gradient Boosting are compared with two baselines: *naive* (tomorrow = today) and *drift* (average daily return). Each model is trained on price-only, price + sentiment, and price + sentiment + event features. This shows whether news adds information.
5. **Evaluation.** An expanding-window walk-forward backtest: the model is trained only on the past and tested on the next day, then the window rolls forward. Reported metrics are MAE, RMSE and MAPE (₹/kg), the direction hit-rate, the percentage MAE improvement over the naive baseline, and a Diebold-Mariano test of whether that difference is statistically significant. For multi-day horizons, training stops `h` days before each test day so that no target overlaps the future.
6. **Forecast.** The model is retrained on all history and rolled forward day by day. An approximate 95% band comes from the backtest errors and widens with the horizon.

## Quick start

```bash
git clone -b polish https://github.com/AAdii-15/MetalMind.git
cd MetalMind
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

metalmind doctor                                   # which data sources work from this machine?
metalmind backfill-news --since 2021-01-01         # rebuild ~5 years of headlines (resumable)
metalmind experiment                               # full study: 4 metals x 11 models x 2 horizons
metalmind forecast --metal copper                  # 10-day forecast, CSV and chart in outputs/
metalmind collect-news                             # run daily to keep the news archive current
```

**Price sources.** The default is `--source yahoo`: international benchmark futures (COMEX copper, COMEX aluminium, US HRC steel, SGX iron ore 62% Fe) converted to ₹/kg with the daily USD/INR rate, giving 5+ years of daily history. These are global benchmarks in rupees, not Indian dealer prices, which add duty, freight and premiums but move with them. You can also pass your own history with `--prices-csv` (a date column and a price column; `2025-07-15` and `15/07/2025` formats both work). BankBazaar, the v1 source, no longer serves its price pages.

**The experiment.** `metalmind experiment` backtests, for each metal and for 1-day and 5-day horizons:
two baselines (naive no-change, drift) and Ridge regression, Random Forest and Gradient Boosting, each trained on three feature sets: *price*, *price + sentiment*, *price + sentiment + events*. It also runs an event study that compares next-day returns after weather, supply, policy, demand, macro and inventory news with days without such news. Everything is written to `results/experiment_<timestamp>/`: `report.md` (all tables), `summary.csv`, `event_study.csv`, `feature_importance.csv`, `data_summary.csv`, per-day predictions and charts.

Other options: `--no-news` (price features only), `--min-train`, `--refit-every`, `--horizons 1 5 10`. Run `metalmind <command> --help` for everything.

## Example output

```text
Walk-forward backtest: 150 test days, next-day horizon

                           MAE (₹/kg)  RMSE (₹/kg)  MAPE (%)  Direction hit-rate (%)  MAE vs naive (%)
Naive (no change)                 ...          ...       ...                     NaN              0.00
Drift (mean return)               ...          ...       ...                     ...               ...
Random Forest: price only         ...          ...       ...                     ...               ...
Random Forest: price + news       ...          ...       ...                     ...               ...
```

A negative *MAE vs naive* means the model beats the no-change baseline. Day-to-day metal prices are close to a random walk, so beating the naive baseline is hard. A model that cannot beat it should not be used for decisions, and the backtest exists to show exactly that.

## Project structure

```
metalmind/
  config.py      metals, keyword rules, news feeds
  prices.py      CSV loader, BankBazaar scraper, price archive
  news.py        RSS collection, Google News backfill, de-duplication, VADER sentiment, trading-day alignment
  events.py      rule-based event tags (weather, supply, policy, demand, macro, inventory)
  features.py    leakage-free features and next-day return target
  models.py      naive and drift baselines, Ridge, Random Forest, Gradient Boosting
  evaluate.py    walk-forward backtest, metrics, Diebold-Mariano test
  experiment.py  the full research grid, event study and report
  forecast.py    recursive forecast with error band
  plotting.py    forecast and backtest charts
  cli.py         command-line interface
tests/           offline tests (synthetic fixtures; no network needed)
notebooks/       MetalMind_prototype_2025.ipynb: the original Colab prototype
```

## Tests

```bash
pytest
```

The tests check keyword matching, de-duplication, weekend-news alignment and the absence of look-ahead leakage. They confirm that walk-forward training only ever sees the past and that the metrics are correct, and they run an end-to-end CLI test.

## Limitations

- Yahoo prices are international benchmarks converted to rupees, not Indian dealer prices.
- Google News search results are a sample of coverage, not every article ever published.
- VADER is a general-purpose sentiment model. Commodity language can mislead it: "prices fall" reads as negative, but it is good news for a buyer. A finance-tuned model such as FinBERT is a natural next step.
- Future news is unknown, so the multi-day forecast carries the latest sentiment forward, and uncertainty grows quickly with the horizon.
- Forecasts are research estimates, not guaranteed prices or investment advice.

## Version history

- **v2.0 (October 2026):** refactored into a tested Python package with a CLI. Adds real-data-only price loading with archives, a daily news archive, word-boundary keyword rules, leakage-free return features, walk-forward evaluation against baselines, a price-only vs price + news comparison, and forecast error bands.
- **v1.0 (July 2025):** Colab notebook prototype (`notebooks/MetalMind_prototype_2025.ipynb`). It used BankBazaar prices, RSS headlines, VADER and a Random Forest on % change and sentiment. It had no backtest, and when scraping failed it filled the history with simulated prices.

## Roadmap

- Finance-tuned sentiment (FinBERT) and event tags (supply disruption, policy, macro releases)
- Longer histories from exchange data (MCX / LME) and INR/USD as a feature
- Gradient boosting and sequence models, compared in the same walk-forward framework
- Simple dashboard for procurement teams

## Author

**Adityaraj Dhuria**, B.Tech (AI/ML) · Intern, Shri Ram Products

Feedback and collaboration are welcome. Please open an issue.
