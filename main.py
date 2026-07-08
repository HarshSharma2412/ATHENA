"""ATHENA AI Investment Research Platform - single entry point.

Run with Streamlit:

    streamlit run main.py

Or directly (imports only; Streamlit provides the runtime):

    python main.py
"""
from __future__ import annotations

from athena.app.dashboard import main

if __name__ == "__main__":
    main()
