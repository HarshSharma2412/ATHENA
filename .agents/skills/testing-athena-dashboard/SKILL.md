---
name: testing-athena-dashboard
description: Test the ATHENA Streamlit dashboard end-to-end (company autocomplete search, summary cards, financial ratios, Plotly charts, financial tables). Use when verifying dashboard/UI or search-engine changes.
---

# Testing the ATHENA dashboard

## What the app is
A Streamlit investment-research dashboard. Real code historically lives on tag `v0.1` (base branch `work-v01`), not `main`. Uses **live yfinance data** (Yahoo Finance) for NSE tickers — `FinancialService.normalize_ticker` appends `.NS`.

## Run it locally
```bash
cd <repo root>
python -m venv .venv && source .venv/bin/activate   # first time
pip install -r requirements.txt
streamlit run main.py --server.port 8501 --server.headless true
```
The single entry point is `main.py` at the repo root (it calls `athena.app.dashboard.main`). No `PYTHONPATH` is needed — the repo root is on `sys.path` and everything imports as `athena.*`.

## Verify the data pipeline WITHOUT the UI (fast sanity check)
Live network to Yahoo is available from the VM. Confirm ratios/formatting before spinning up the browser:
```bash
python -c "
import athena.app.dashboard as d
snap = d._load_company_snapshot('TCS')
print(d._build_summary_payload(snap))
print(d._build_ratio_payload(snap))
print({k: list(v['year']) for k,v in d._build_chart_data(snap).items() if hasattr(v,'columns') and 'year' in getattr(v,'columns',[])})
"
```
If yfinance ever returns empty frames (rate limiting / network), UI testing will show 'No data' — retry or note it.

## UI test flow (record this)
1. Sidebar has a **text input** + a **Matches selectbox**. The text input is a Streamlit `st.text_input`: it only applies on **Enter/blur**, NOT per keystroke. So type the query then press `Return`, then open the Matches dropdown.
2. Good adversarial queries: `hc` (→ HCLTECH/HCC/HCG/HCL-INSYS/HDFCBANK), `waa` (→ WAAREEENER first), `tata` (→ Tata group grouped by company name, incl. TCS/TMPV). Dropdown rows are formatted `TICKER — Company Name · Sector`.
3. Selecting a match **auto-loads** the dashboard (no Search button exists).
4. Check summary cards: `₹` prices with Indian grouping, Market Cap in `Cr`, PE/PB 2-dp, Dividend Yield often `N/A`.
5. Check ratios never show `nan` — they use `%`/numbers or `N/A`.
6. Charts: Revenue/PAT/FCF are Plotly line+marker, x-axis = years, y-axis = `₹ Crore`; hover shows `₹… Cr`.
7. Financial tables: only important metrics, years as columns.

## Known cosmetic issues (verify if fixed; don't re-flag as new)
- Company card may show the raw `.NS` ticker (e.g. `WAAREEENER.NS`) when yfinance lacks a long name.
- Financial-table empty cells may render `None` instead of `N/A`.

## Recording tips
- Maximize Chrome first: `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`.
- Streamlit reruns on every widget change; give it ~2-4s after Enter/selection before screenshotting.
- Use the `zoom` action to read dropdown rows and chart tooltips clearly.

## Devin Secrets Needed
None. Uses public NSE archives (for the company master build) and public Yahoo Finance data — no API keys.
