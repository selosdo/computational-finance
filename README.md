# Computational Finance in Python — Lecture 02 Assessment

Three-signal long-only equity strategy on AAPL · MSFT · AMZN with the S&P 500
as a benchmark. Course: *Computational Finance in Python*, Universität
Tübingen, SoSe 2026.

## Files

| File | Purpose |
|---|---|
| `assessment_notebook.ipynb` | Client-facing wrap-up: three signals, statistics, plots. Predefined cells preserved verbatim. |
| `research_notebook.ipynb` | Empirical evidence: parameter calibration on 2010–2018, out-of-sample validation on 2019–2025, robustness, bootstrap CIs. |
| `module.py` | All reusable code — NumPy-only signal generators, performance metrics, backtest helpers, data loader with offline cache. |
| `report.tex` | Standalone LaTeX write-up with theory, methodology, results and references (compile with `pdflatex report.tex`). |
| `data_cache/prices.csv` | Offline price cache — lets the notebooks run without network. |
| `requirements.txt` | Pinned package list. |

## Reproducing the results

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run both notebooks end-to-end (the cache means no network is required)
jupyter nbconvert --to notebook --execute research_notebook.ipynb   --output research_notebook.executed.ipynb
jupyter nbconvert --to notebook --execute assessment_notebook.ipynb --output assessment_notebook.executed.ipynb

# 3. (Optional) compile the LaTeX write-up
pdflatex report.tex
```

If `yahooquery` is unavailable or no network is reachable, both notebooks
fall back to `data_cache/prices.csv` automatically.

## Headline result

| Metric | Strategy | S&P 500 buy-and-hold |
|---|---:|---:|
| CAGR | 10.6 % | 12.0 % |
| Annualised vol. | 10.0 % | 17.3 % |
| Sharpe | **1.06** (95 % CI [0.53, 1.63]) | 0.69 |
| Max drawdown | **−12.4 %** | −33.9 % |
| Calmar | **0.86** | 0.35 |

Period: 2010-01-04 → 2025-12-30, 4 022 trading days, 78 trades.
