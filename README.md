# ATHENA

AI-powered Equity Research Platform for long-term investors.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

`main.py` is the single entry point; it calls `athena.app.dashboard.main()`.
No `PYTHONPATH` is required — everything imports as `athena.*`.

Refresh the NSE company master (optional):

```bash
python athena/scripts/update_company_master.py
```

## Architecture

Clean, layered structure — each package has a single responsibility:

```
athena/
  app/         Streamlit UI only (dashboard, components, pages)
  services/    Data acquisition & parsing (no calculations)
  engines/     Calculations (ratio, dcf, forecast)
  models/      Dataclasses / value objects
  database/    Persistence (SQLAlchemy)
  charts/      Reusable chart builders
  utils/       Formatting, math and helper utilities
  config/      Configuration
  ai/          AI providers & prompts
  data/        Static data (company_master.csv)
  tests/       Test suite
```

## Development

```bash
python -m pytest      # tests
ruff check .          # lint
mypy athena           # type-check
```
