"""
module.py
---------
Reusable building blocks for the Lecture-02 assessment.

Design rules (from the assessment hints):
  * Numerical computations use NumPy only — no `pd.rolling().mean()` etc.
    Pandas is used only as a labelled container so the predefined
    assessment-notebook cells continue to work unchanged.
  * Functions are written to be called from BOTH notebooks
    (assessment + research) without copy-paste.
  * Names are self-explanatory; comments describe *why*, not *what*.

Sections:
  1. Data loading (yahooquery -> local CSV cache fallback)
  2. NumPy primitives  (SMA, EMA, rolling std, RSI, momentum)
  3. Signal generators (SMA crossover, RSI mean-reversion, 12-1 momentum)
  4. Performance metrics
  5. Backtesting helpers (single-asset wealth path, parameter grid search,
     train/test split, bootstrap CI)
"""

import os
import numpy as np
import pandas as pd

# yahooquery is optional: if unavailable we fall back to a CSV cache so the
# notebooks still run on a fresh machine with no network.
try:
    from yahooquery import Ticker as _YahooTicker
    _YAHOOQUERY_AVAILABLE = True
except Exception:
    _YAHOOQUERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def _default_cache_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')


def download_stock_price_data(tickers, start_date, end_date, cache_dir=None):
    """
    Adjusted-close prices and 1-day multiplicative ratios for `tickers`.

    Strategy:
      1. Try a live `yahooquery` download. On success, refresh the local cache.
      2. If yahooquery is missing or the request fails, fall back to the
         shipped CSV cache. This keeps the notebooks runnable offline.

    Returns
    -------
    df_prices : pd.DataFrame  (dates x tickers)
        Adjusted-close prices.
    df_price_changes : pd.DataFrame  (dates x tickers)
        Daily price ratios price[t]/price[t-1]; the first row is 1.
        This matches the contract expected by the predefined
        capital-allocation cell of the assessment notebook.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    cache_file = os.path.join(cache_dir, 'prices.csv')

    df_prices = None
    last_error = None

    # Try live download first so the cache stays fresh.
    if _YAHOOQUERY_AVAILABLE:
        try:
            raw = _YahooTicker(tickers).history(start=start_date, end=end_date)
            if not isinstance(raw, pd.DataFrame) or 'adjclose' not in raw.columns:
                raise RuntimeError('yahooquery did not return adjusted-close data')
            df = raw['adjclose'].unstack(level=0)
            df.index = pd.to_datetime(df.index)
            df = df.reindex(columns=tickers).dropna()
            if len(df) > 0:
                df_prices = df
                os.makedirs(cache_dir, exist_ok=True)
                # Merge with existing cache so previously-downloaded tickers
                # remain available even if they weren't requested this time.
                if os.path.exists(cache_file):
                    cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    combined = df_prices.combine_first(cached)
                else:
                    combined = df_prices
                combined.sort_index().to_csv(cache_file)
        except Exception as exc:
            last_error = exc

    # Fall back to the cache if the live download did not produce data.
    if df_prices is None:
        if not os.path.exists(cache_file):
            raise RuntimeError(
                f'No data available: yahooquery available={_YAHOOQUERY_AVAILABLE}; '
                f'last download error={last_error!r}; no cache at {cache_file}. '
                f'Run once with network access to populate the cache.'
            )
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in cached.columns]
        if missing:
            raise RuntimeError(
                f'Tickers {missing} not present in the cache. '
                f'Run once online to download them.'
            )
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df_prices = cached.loc[(cached.index >= start) & (cached.index < end), tickers].dropna()

    # Multiplicative daily ratios; first row defined as 1 so the wealth path
    # is well defined from day 0.
    prev = df_prices.to_numpy(dtype=float)
    ratios = np.ones_like(prev)
    ratios[1:] = prev[1:] / prev[:-1]
    df_price_changes = pd.DataFrame(ratios, index=df_prices.index, columns=df_prices.columns)
    return df_prices, df_price_changes


# ---------------------------------------------------------------------------
# 2. NumPy primitives
# ---------------------------------------------------------------------------

def simple_moving_average(prices, window):
    """SMA via cumulative sums; the first ``window-1`` entries are NaN.

    The cumsum trick is O(n) instead of O(n*window) and avoids the
    boundary-zero artefact of ``np.convolve(..., mode='same')``.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    out = np.full(n, np.nan)
    if window <= 0 or window > n:
        return out
    cs = np.cumsum(np.insert(prices, 0, 0.0))
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    return out


def exponential_moving_average(prices, span):
    """Standard EMA with smoothing factor ``alpha = 2 / (span + 1)``.

    Implemented as an explicit recursion to keep every numerical step visible.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    out = np.full(n, np.nan)
    if n == 0 or span < 1:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = prices[0]
    for t in range(1, n):
        out[t] = alpha * prices[t] + (1.0 - alpha) * out[t - 1]
    return out


def rolling_std(prices, window):
    """Population rolling standard deviation via cumulative sums of x and x^2."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    out = np.full(n, np.nan)
    if window < 2 or window > n:
        return out
    cs = np.cumsum(np.insert(prices, 0, 0.0))
    cs2 = np.cumsum(np.insert(prices ** 2, 0, 0.0))
    sums = cs[window:] - cs[:-window]
    sums2 = cs2[window:] - cs2[:-window]
    means = sums / window
    var = sums2 / window - means ** 2
    var = np.maximum(var, 0.0)  # guard against tiny negative roundoff
    out[window - 1:] = np.sqrt(var)
    return out


def relative_strength_index(prices, window=14):
    """Wilder's RSI.

    Wilder uses an SMA seed for the first window of gains/losses, then a
    recursive smoother (avg' = (avg * (window - 1) + new) / window). Returns
    NaN for the first ``window`` observations.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    rsi = np.full(n, np.nan)
    if n < window + 1 or window < 1:
        return rsi
    deltas = np.diff(prices)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = float(np.sum(gains[:window]) / window)
    avg_loss = float(np.sum(losses[:window]) / window)
    rsi[window] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for t in range(window + 1, n):
        avg_gain = (avg_gain * (window - 1) + gains[t - 1]) / window
        avg_loss = (avg_loss * (window - 1) + losses[t - 1]) / window
        rsi[t] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def momentum(prices, lookback=252, skip=21):
    """12-1 style time-series momentum: return from t-lookback to t-skip.

    With the defaults (lookback=252 trading days ~ 12 months,
    skip=21 trading days ~ 1 month) this skips the most recent month to
    avoid contaminating the trend signal with short-term reversal.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    mom = np.full(n, np.nan)
    if n <= lookback or skip < 0 or skip >= lookback:
        return mom
    t = np.arange(lookback, n)
    numerator = prices[t - skip]
    denominator = prices[t - lookback]
    valid = denominator > 0.0
    mom[t[valid]] = numerator[valid] / denominator[valid] - 1.0
    return mom


# ---------------------------------------------------------------------------
# 3. Signal generators
#
# Contract (matches the predefined assessment-notebook cells):
#   input  : pd.Series of prices indexed by date
#   output : pd.DataFrame with the same index and columns
#            'signal'           in {0, 1}        (long/flat indicator)
#            'position_change'  in {-1, 0, +1}   (np.diff of signal,
#                                                 first entry forced to 0)
# We never emit -1 without a preceding +1 — guaranteed by construction
# because position_change is the diff of a {0,1} sequence.
# ---------------------------------------------------------------------------

def _signal_dataframe(signal_array, index):
    """Wrap a 0/1 signal array into the (signal, position_change) DataFrame."""
    signal_array = np.asarray(signal_array, dtype=float)
    position_change = np.zeros_like(signal_array)
    position_change[1:] = signal_array[1:] - signal_array[:-1]
    # Defensive: the first row must not look like a trade close.
    position_change[0] = 0.0
    return pd.DataFrame(
        {'signal': signal_array, 'position_change': position_change},
        index=index,
    )


def ma_crossover_signal(series, short_window, long_window):
    """SMA crossover: long when SMA(short) > SMA(long), flat otherwise.

    Reference: Brock, Lakonishok & LeBaron (1992); Marshall, Nguyen &
    Visaltanachoti (2017).
    """
    if short_window >= long_window:
        raise ValueError('short_window must be strictly less than long_window')
    prices = series.to_numpy(dtype=float)
    short_ma = simple_moving_average(prices, short_window)
    long_ma = simple_moving_average(prices, long_window)
    valid = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    signal = np.zeros_like(prices)
    signal[valid] = (short_ma[valid] > long_ma[valid]).astype(float)
    return _signal_dataframe(signal, series.index)


def rsi_signal(series, window=14, lower=30.0, upper=70.0):
    """RSI mean-reversion with hysteresis.

    State machine, long-only:
      * flat -> long  when RSI crosses BELOW `lower` (oversold)
      * long -> flat  when RSI crosses ABOVE `upper` (overbought)
    Held in between so we don't churn around the thresholds.

    Reference: Wilder (1978); Chong & Ng (2008).
    """
    if not (0.0 < lower < upper < 100.0):
        raise ValueError('require 0 < lower < upper < 100')
    prices = series.to_numpy(dtype=float)
    rsi = relative_strength_index(prices, window=window)
    signal = np.zeros_like(prices)
    state = 0
    for t in range(len(prices)):
        if not np.isnan(rsi[t]):
            if state == 0 and rsi[t] < lower:
                state = 1
            elif state == 1 and rsi[t] > upper:
                state = 0
        signal[t] = state
    return _signal_dataframe(signal, series.index)


def momentum_signal(series, lookback=252, skip=21, threshold=0.0):
    """12-1 time-series momentum: long when past 12-1m return > threshold.

    Reference: Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012).
    """
    prices = series.to_numpy(dtype=float)
    mom = momentum(prices, lookback=lookback, skip=skip)
    signal = np.zeros_like(prices)
    valid = ~np.isnan(mom)
    signal[valid] = (mom[valid] > threshold).astype(float)
    return _signal_dataframe(signal, series.index)


# ---------------------------------------------------------------------------
# 4. Performance metrics  (all NumPy, all ``axis=0``-aware where useful)
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices):
    """Simple daily returns r_t = p_t / p_{t-1} - 1; length n-1."""
    prices = np.asarray(prices, dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def equity_curve_from_returns(returns, initial=1.0):
    """Wealth path starting at `initial` and compounding by (1 + r_t)."""
    returns = np.asarray(returns, dtype=float)
    return initial * np.cumprod(1.0 + returns)


def annualized_return(returns, periods=TRADING_DAYS_PER_YEAR):
    """Geometric annualised return."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.nan
    total = float(np.prod(1.0 + returns))
    if total <= 0.0:
        return -1.0
    return total ** (periods / len(returns)) - 1.0


def annualized_volatility(returns, periods=TRADING_DAYS_PER_YEAR):
    """Annualised population std of daily returns."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return np.nan
    mean = float(np.sum(returns) / len(returns))
    var = float(np.sum((returns - mean) ** 2) / len(returns))
    return np.sqrt(var * periods)


def sharpe_ratio(returns, periods=TRADING_DAYS_PER_YEAR, risk_free=0.0):
    """Annualised Sharpe ratio with a constant risk-free rate."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return np.nan
    ann_ret = annualized_return(returns, periods)
    ann_vol = annualized_volatility(returns, periods)
    if ann_vol == 0.0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free) / ann_vol


def sortino_ratio(returns, periods=TRADING_DAYS_PER_YEAR, target=0.0):
    """Annualised Sortino ratio: excess-return over downside semi-deviation."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return np.nan
    daily_target = target / periods
    downside = np.minimum(returns - daily_target, 0.0)
    dd_var = float(np.sum(downside ** 2) / len(returns))
    dd_vol = np.sqrt(dd_var * periods)
    if dd_vol == 0.0:
        return np.nan
    return (annualized_return(returns, periods) - target) / dd_vol


def max_drawdown(equity_curve):
    """Most negative drawdown observed; returns a non-positive number."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) == 0:
        return np.nan
    peaks = np.maximum.accumulate(eq)
    return float(np.min((eq - peaks) / peaks))


def calmar_ratio(returns, periods=TRADING_DAYS_PER_YEAR):
    """CAGR divided by absolute max drawdown."""
    returns = np.asarray(returns, dtype=float)
    eq = equity_curve_from_returns(returns)
    mdd = max_drawdown(eq)
    if mdd == 0.0 or np.isnan(mdd):
        return np.nan
    return annualized_return(returns, periods) / abs(mdd)


def hit_rate(returns):
    """Share of strictly-positive return days among non-zero days."""
    returns = np.asarray(returns, dtype=float)
    nonzero = returns[returns != 0.0]
    if len(nonzero) == 0:
        return np.nan
    return float(np.sum(nonzero > 0.0) / len(nonzero))


def number_of_trades(position_change):
    """Number of buy events (= +1 in the position_change column)."""
    pc = np.asarray(position_change, dtype=float)
    return int(np.sum(pc > 0.5))


def average_holding_days(signal_array):
    """Mean length, in trading days, of contiguous in-the-market spells."""
    sig = np.asarray(signal_array, dtype=float)
    if len(sig) == 0:
        return np.nan
    diffs = np.diff(np.concatenate(([0.0], sig, [0.0])))
    starts = np.where(diffs > 0.5)[0]
    ends = np.where(diffs < -0.5)[0]
    if len(starts) == 0:
        return np.nan
    return float(np.mean(ends - starts))


def time_in_market(signal_array):
    """Fraction of days the strategy is long."""
    sig = np.asarray(signal_array, dtype=float)
    if len(sig) == 0:
        return np.nan
    return float(np.sum(sig > 0.5) / len(sig))


def performance_summary(returns, signal_array=None, position_change=None,
                        periods=TRADING_DAYS_PER_YEAR):
    """One-stop dictionary of headline metrics; useful for tables."""
    eq = equity_curve_from_returns(returns)
    out = {
        'total_return': float(eq[-1] - 1.0) if len(eq) else np.nan,
        'cagr': annualized_return(returns, periods),
        'ann_vol': annualized_volatility(returns, periods),
        'sharpe': sharpe_ratio(returns, periods),
        'sortino': sortino_ratio(returns, periods),
        'calmar': calmar_ratio(returns, periods),
        'max_drawdown': max_drawdown(eq),
        'hit_rate': hit_rate(returns),
    }
    if signal_array is not None:
        out['time_in_market'] = time_in_market(signal_array)
        out['avg_holding_days'] = average_holding_days(signal_array)
    if position_change is not None:
        out['n_trades'] = number_of_trades(position_change)
    return out


# ---------------------------------------------------------------------------
# 5. Backtesting helpers
# ---------------------------------------------------------------------------

def backtest_long_flat(prices, signal_array, transaction_cost_bps=0.0):
    """Single-asset wealth path with a 1-day execution lag.

    The investor sees signal[t] at end-of-day t and trades at the next open;
    so the day-(t+1) return is multiplied by signal[t]. This matches the
    timing assumption used by the predefined multi-asset allocator in the
    assessment notebook (which acts on `position_change` observed at day t).

    Parameters
    ----------
    prices : 1-D array
    signal_array : 1-D array of 0/1 values, same length as prices
    transaction_cost_bps : round-trip transaction cost in basis points

    Returns
    -------
    strategy_returns : 1-D array of length n-1
    """
    prices = np.asarray(prices, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    if len(prices) != len(sig):
        raise ValueError('prices and signal_array must have equal length')
    rets = prices[1:] / prices[:-1] - 1.0
    pos = sig[:-1]
    strat = pos * rets
    if transaction_cost_bps > 0.0:
        # cost is charged the day a position is opened or closed
        prev_pos = np.concatenate(([0.0], pos[:-1]))
        traded = np.abs(pos - prev_pos)
        strat = strat - traded * (transaction_cost_bps / 10_000.0)
    return strat


def parameter_grid_search(price_panel, signal_fn, param_grid,
                          score_fn=None, periods=TRADING_DAYS_PER_YEAR):
    """Evaluate a signal across (parameters x tickers) and rank by mean score.

    Parameters
    ----------
    price_panel : dict[str, pd.Series]
        Mapping ticker -> price series.
    signal_fn : callable(pd.Series, **params) -> pd.DataFrame with 'signal'
    param_grid : list of dicts
        Each dict is one combination of keyword arguments to ``signal_fn``.
    score_fn : callable(returns_array) -> float
        Default: annualised Sharpe.

    Returns
    -------
    list of dicts {'params', 'mean_score', 'median_score', 'per_ticker'}
    """
    if score_fn is None:
        score_fn = lambda r: sharpe_ratio(r, periods)
    results = []
    for params in param_grid:
        per_ticker = {}
        for ticker, series in price_panel.items():
            try:
                sig_df = signal_fn(series, **params)
                strat = backtest_long_flat(
                    series.to_numpy(dtype=float),
                    sig_df['signal'].to_numpy(dtype=float),
                )
                per_ticker[ticker] = float(score_fn(strat))
            except Exception:
                per_ticker[ticker] = np.nan
        scores = np.array([s for s in per_ticker.values() if not np.isnan(s)])
        results.append({
            'params': params,
            'mean_score': float(np.mean(scores)) if len(scores) else np.nan,
            'median_score': float(np.median(scores)) if len(scores) else np.nan,
            'per_ticker': per_ticker,
        })
    return results


def train_test_split_by_date(df, split_date):
    """Chronological split. ``split_date`` becomes the first OOS date."""
    split = pd.to_datetime(split_date)
    return df.loc[df.index < split], df.loc[df.index >= split]


def bootstrap_sharpe_ci(returns, n_boot=10_000, alpha=0.05,
                        periods=TRADING_DAYS_PER_YEAR, seed=42):
    """IID bootstrap confidence interval for the annualised Sharpe ratio.

    We resample the daily-return series with replacement; this destroys serial
    dependence (which is small for daily equity returns) but gives a
    distribution-free CI without scipy.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        sample = returns[rng.integers(0, n, size=n)]
        sharpes[b] = sharpe_ratio(sample, periods)
    sharpes = sharpes[~np.isnan(sharpes)]
    if len(sharpes) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(sharpes, alpha / 2.0)),
        float(np.quantile(sharpes, 1.0 - alpha / 2.0)),
    )


def _erf_approx(x):
    """Abramowitz-Stegun 7.1.26 approximation to erf, max error ~ 1.5e-7."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.where(x < 0.0, -1.0, 1.0)
    z = np.abs(np.asarray(x, dtype=float))
    tt = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * np.exp(-z * z)
    return sign * y


def _two_sided_normal_pvalue(t_stat):
    """Two-sided p-value under the standard normal: 2 * (1 - Phi(|t|))."""
    z = abs(float(t_stat))
    return float(2.0 * (1.0 - 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))))


def paired_t_test_returns(returns_a, returns_b):
    """Paired t-statistic on two daily-return series (same length).

    Returns (t_stat, two_sided_p_value) using a normal-approximation
    p-value (acceptable here because n >> 30 in our backtests).
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return np.nan, np.nan
    diff = a - b
    mean = float(np.sum(diff) / len(diff))
    var = float(np.sum((diff - mean) ** 2) / (len(diff) - 1))
    se = np.sqrt(var / len(diff))
    if se == 0.0:
        return np.nan, np.nan
    t = mean / se
    return float(t), _two_sided_normal_pvalue(t)
