"""MetalMind: news-sentiment-aware price forecasting for industrial metals.

The package is organised as a small pipeline:

    prices   -> load a daily price history (CSV or BankBazaar)
    news     -> collect metal-related headlines from RSS and score sentiment
    features -> build leakage-free features aligned to trading days
    models   -> baselines and the Random Forest return model
    evaluate -> walk-forward backtest (MAE / RMSE / MAPE / direction)
    forecast -> recursive multi-day forecast with an error band
"""

__version__ = "2.0.0"
