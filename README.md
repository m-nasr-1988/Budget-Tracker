# Budget-Tracker
Streamlit App for Budget Tracker

Run locally:
python -m venv .venv && source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
Open the local URL shown in terminal
Deploy (Streamlit Cloud):

Push app.py and requirements.txt to your GitHub repo
Go to share.streamlit.io (or streamlit.io/cloud), “New app”
Select your repo and branch, set app path to app.py, Deploy
Share the resulting link
Optional sample file:

Add sample_data/sample.xlsx with the columns Month, Type, Source/Category, Amount, Status as per your example. Use the Import tab to upload.
How it meets your requirements

Edit/delete entries: Entries tab includes forms to update or delete by ID.
Upload Excel with entries: Import tab supports Excel/CSV; maps columns; previews and imports.
Automatic categorization for non-manual expenses: Rule engine + keyword heuristics; matches to manual entries by amount within the same month first.
Add uploaded data into table: Imported rows become entries tied to a new import batch.
Copy from a saved generic template: Create templates and apply to any month.
Save a template for entries: Save current month entries as template with one click.
Month selector: Sidebar month selection with per-month summarization.
Charts for insights: Dashboard includes category donut, 6-month trends, KPIs, budgets progress, and a simple forecast.
Example entries: One-click seed for June with your provided example.
Extra goodies

Budgets by category with progress bars
Tags and Notes on entries
CSV export
Rule management with priorities and optional regex
Simple next-month forecast (average of last 3 months)
Status options (Paid, Pending, Planned)
Questions to tailor it for you

Are you happy using Streamlit (fast deploy + shareable link)? If you prefer Flask/FastAPI with a custom UI, I can provide that instead.
Do you want authentication (multi-user) and/or cloud DB (e.g., Supabase/Neon) for persistence across redeploys?
Any custom category list or currency preferences?
Should we add recurring auto-apply templates per month or alerts when budgets are exceeded?
