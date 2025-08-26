# Budget Tracker

Streamlit + SQLite-based budget planner.
- Add/Edit/Delete entries
- Upload Excel/CSV and auto-categorize
- Templates, budgets, charts, export

## Run locally
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Cloud)
Push this repo to GitHub, then deploy at https://streamlit.io/cloud (New app → select repo → app.py).
