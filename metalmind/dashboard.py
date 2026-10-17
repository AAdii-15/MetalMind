"""METAL MIND dashboard (Streamlit).

    metalmind dashboard            # or: streamlit run metalmind/dashboard.py

Pick a metal to see the latest benchmark price in INR/kg, a forecast with its
backtested error band, the latest tagged news with sentiment, and how every
model compares with the naive "no change" baseline. "Offline mode" uses the
last saved download, so a demo still works without internet.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from metalmind import evaluate, news, plotting, prices
from metalmind.config import get_metal
from metalmind.events import tag
from metalmind.forecast import forecast

DATA_DIR = Path(os.environ.get("METALMIND_DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("METALMIND_RESULTS_DIR", "results"))
START = "2021-01-01"
LABELS = {"copper": "Copper", "aluminium": "Aluminium", "steel": "Steel (HRC)", "iron_ore": "Iron ore"}
RIDGE = "Ridge regression: price + news"

st.set_page_config(page_title="METAL MIND", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Loading prices...")
def load_prices(metal: str, offline: bool) -> tuple[pd.Series, str]:
    return prices.get_price_history(metal, data_dir=DATA_DIR, source="cache" if offline else "yahoo", start=START)


@st.cache_data(ttl=3600, show_spinner="Aligning news with trading days...")
def load_news(metal: str, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    series, _ = load_prices(metal, offline)
    archive = news.load_archive(metal, DATA_DIR)
    return archive, news.daily_signals(archive, series.index)


@st.cache_data(ttl=3600, show_spinner="Running the walk-forward backtest (about 20 seconds)...")
def backtest(metal: str, offline: bool) -> evaluate.Backtest:
    series, _ = load_prices(metal, offline)
    _, signals = load_news(metal, offline)
    return evaluate.run(series, signals, min_train=250, refit_every=20, horizon=1)


@st.cache_data(ttl=3600, show_spinner="Forecasting...")
def run_forecast(metal: str, offline: bool, horizon: int) -> pd.DataFrame:
    series, _ = load_prices(metal, offline)
    _, signals = load_news(metal, offline)
    p = backtest(metal, offline).predictions
    resid = float(np.std((p[RIDGE] - p["actual"]) / p["last_price"]))
    return forecast(series, signals, horizon=horizon, residual_std=resid)


def latest_experiment() -> Path | None:
    runs = sorted(glob.glob(str(RESULTS_DIR / "experiment_*" / "summary.csv")))
    return Path(runs[-1]).parent if runs else None


# ------------------------------------------------------------------ sidebar
st.sidebar.title("METAL MIND")
st.sidebar.caption("News-sentiment-aware metal price research tool. Shri Ram Products, R&D.")
metal = st.sidebar.selectbox("Metal", list(LABELS), format_func=LABELS.get)
horizon = st.sidebar.slider("Forecast horizon (trading days)", 5, 20, 10)
history_days = st.sidebar.slider("History shown (trading days)", 60, 500, 120, step=20)
offline = st.sidebar.toggle("Offline mode (use saved data)", value=os.environ.get("METALMIND_OFFLINE") == "1",
                            help="Uses the last successful download, so the demo works without internet.")
if st.sidebar.button("Fetch latest news", disabled=offline, width="stretch"):
    with st.spinner("Collecting headlines..."):
        _, new = news.collect(metal, DATA_DIR)
    st.cache_data.clear()
    st.sidebar.success(f"{new} new articles added")

# ------------------------------------------------------------------ data
try:
    series, source = load_prices(metal, offline)
except prices.PriceDataError as exc:
    st.error(f"Could not load prices: {exc}")
    st.stop()
archive, signals = load_news(metal, offline)
bt = backtest(metal, offline)
p = bt.predictions
fc = run_forecast(metal, offline, horizon)

# ------------------------------------------------------------------ header
spec = get_metal(metal)
st.title(f"{LABELS[metal]}: price outlook and news monitor")
st.caption(f"Price source: {source}. Benchmark: {spec.benchmark} converted to ₹/kg. "
           "Forecasts are research estimates with measured error, not guaranteed prices.")

last, prev = float(series.iloc[-1]), float(series.iloc[-2])
end = float(fc["forecast"].iloc[-1])
row = bt.summary.loc[RIDGE]
today = signals.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Latest price ({series.index[-1]:%d %b %Y})", f"₹{last:,.2f}/kg", f"{last / prev - 1:+.2%} vs previous day")
c2.metric(f"Forecast in {horizon} trading days", f"₹{end:,.2f}/kg", f"{end / last - 1:+.2%}", delta_color="off")
c3.metric(f"Ridge error vs naive, p = {row['DM p-value']:.2f}", f"{row['MAE vs naive (%)']:+.1f}%",
          help="Walk-forward backtest. Negative = more accurate than 'price stays the same'. "
               "p < 0.05 = statistically significant (Diebold-Mariano).")
c4.metric("Latest-day news", f"{int(today['news_count'])} articles",
          help=f"Average sentiment {today['sentiment']:+.2f} (VADER, −1 to +1)" if today["news_count"] else "No articles")

tab_fc, tab_news, tab_bt, tab_research, tab_how = st.tabs(
    ["Forecast", "News and sentiment", "Backtest", "Research findings", "How it works"])

# ------------------------------------------------------------------ forecast
with tab_fc:
    st.pyplot(plotting.forecast_figure(series, fc, metal, history_days), width="stretch")
    lo, hi = fc["lower"].iloc[-1], fc["upper"].iloc[-1]
    st.markdown(
        f"In **{horizon} trading days** the model's central estimate is **₹{end:,.2f}/kg**, with an approximate "
        f"95% range of **₹{lo:,.2f} – ₹{hi:,.2f}/kg**. The range comes from the model's own errors in the backtest. "
        "Because no model beat the naive baseline significantly, treat the range, not the central line, "
        "as the useful output (e.g., for tender price-escalation margins).")
    table = fc.rename(columns=str.title).copy()
    table.index = table.index.strftime("%a %d %b %Y")
    st.dataframe(table.style.format("₹{:,.2f}"), width="stretch")

# ------------------------------------------------------------------ news
with tab_news:
    if archive.empty:
        st.info("No news archive yet. Run `metalmind backfill-news` or use 'Fetch latest news'.")
    else:
        st.pyplot(plotting.sentiment_figure(signals, days=history_days), width="stretch")
        recent = archive.sort_values("published_utc", ascending=False).head(25).copy()
        recent["events"] = (recent["title"].fillna("") + ". " + recent["summary"].fillna("")).map(
            lambda t: ", ".join(tag(t)) or "–")
        recent["date"] = recent["date_ist"].dt.strftime("%d %b %Y")
        st.subheader("Latest headlines")
        st.dataframe(
            recent[["date", "title", "sentiment", "events", "link"]],
            width="stretch", hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("Headline", width="large"),
                "sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-1, max_value=1, format="%+.2f"),
                "events": "Event tags",
                "link": st.column_config.LinkColumn("Source", display_text="open"),
            })
        st.caption(f"{len(archive):,} relevant articles in the archive. Sentiment: VADER compound score (−1 to +1). "
                   "Event tags: rule-based (weather, supply, policy, demand, macro, inventory).")

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.markdown(f"Walk-forward test on **{len(p):,} trading days** ({p.index[0]:%d %b %Y} to {p.index[-1]:%d %b %Y}): "
                "each day's forecast uses only data available before that day. Next-day horizon.")
    show = bt.summary[["MAE (₹/kg)", "RMSE (₹/kg)", "MAPE (%)", "Direction hit-rate (%)", "MAE vs naive (%)", "DM p-value"]]
    st.dataframe(show.astype(float).sort_values("MAE (₹/kg)").style.format("{:,.3f}", na_rep="–"), width="stretch")
    st.pyplot(plotting.backtest_figure(p.iloc[-min(250, len(p)):], RIDGE, metal), width="stretch")
    st.caption("If the predicted line looks like the actual line shifted by one day, the model is essentially "
               "predicting 'no change', which is the typical behaviour of an efficient, actively traded market.")

# ------------------------------------------------------------------ research findings
with tab_research:
    run = latest_experiment()
    if run is None:
        st.info("Run `metalmind experiment` to add the full study (11 models × 3 feature sets × 2 horizons).")
    else:
        st.caption(f"From the full experiment run {run.name.replace('experiment_', '')} (results folder)")
        summ = pd.read_csv(run / "summary.csv")
        m = summ[summ["metal"] == spec.name]
        for h in sorted(m["horizon (days)"].unique()):
            st.markdown(f"**{h}-day horizon**")
            st.dataframe(m[m["horizon (days)"] == h].set_index("model")[["MAE (₹/kg)", "MAE vs naive (%)", "DM p-value"]]
                         .astype(float).sort_values("MAE (₹/kg)").style.format("{:,.3f}", na_rep="–"), width="stretch")
        ev = run / "event_study.csv"
        if ev.exists():
            st.markdown("**Event study: next-day return after each type of news vs other days**")
            e = pd.read_csv(ev)
            st.dataframe(e[e["metal"] == spec.name].drop(columns="metal").set_index("event")
                         .style.format("{:,.3f}", na_rep="–"), width="stretch")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("""
1. **Collect.** Daily benchmark futures prices from Yahoo Finance are converted to ₹/kg with the USD/INR rate.
   Headlines come from Google News (historical, date-bounded searches), Livemint and Economic Times.
2. **Clean.** Timestamps are converted to IST, duplicate and syndicated stories are removed, and word-boundary keyword rules
   keep only articles about the metal.
3. **Understand.** Each article gets a VADER sentiment score and event tags (weather, supply, policy, demand, macro, inventory).
4. **Align.** Each article is assigned to the first trading day on or after its publication, so weekend news counts towards Monday
   and the model never sees the future.
5. **Model.** Ridge regression (and Random Forest) predict the next return from lagged returns, volatility, momentum,
   sentiment and event counts, and are compared with the naive "no change" forecast in a walk-forward backtest.
6. **Forecast.** The model rolls forward day by day; the shaded band is built from its backtest errors.
""")
