import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
from pathlib import Path
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = Path("budget.db")

# ------------- DB Utils -------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def now_ts():
    return datetime.utcnow().isoformat(timespec="seconds")

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        period TEXT, -- YYYY-MM
        type TEXT CHECK(type IN ('Income','Expense')),
        category TEXT,
        source TEXT,
        amount REAL,
        status TEXT,
        notes TEXT,
        is_manual INTEGER DEFAULT 0,
        import_batch_id INTEGER,
        tags TEXT,
        created_at TEXT,
        updated_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS import_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        imported_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT,
        category TEXT,
        type TEXT,     -- 'Income','Expense','Any'
        field TEXT,    -- 'source','notes','category','any'
        regex INTEGER DEFAULT 0,
        priority INTEGER DEFAULT 100
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS template_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        template_id INTEGER,
        type TEXT,
        category TEXT,
        source TEXT,
        amount REAL,
        status TEXT,
        notes TEXT,
        FOREIGN KEY(template_id) REFERENCES templates(id)
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        period TEXT, -- YYYY-MM
        category TEXT,
        amount REAL
    );
    """)
    conn.commit()
    conn.close()

def month_name_to_num(name: str) -> int:
    months = {m: i for i, m in enumerate([
        "", "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])}
    return months.get(name.strip().capitalize(), 0)

def to_period(dt: date) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"

def ensure_period_from_parts(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"

def existing_periods():
    conn = get_conn()
    q = conn.execute("SELECT DISTINCT period FROM entries ORDER BY period").fetchall()
    conn.close()
    vals = [r["period"] for r in q]
    if not vals:
        # include current + next 11 months as options
        today = date.today()
        vals = [ensure_period_from_parts(today.year, today.month)]
    return vals

def add_entry(entry: dict):
    conn = get_conn()
    cur = conn.cursor()
    entry.setdefault("notes", "")
    entry.setdefault("status", "Pending")
    entry.setdefault("is_manual", 0)
    entry.setdefault("source", "")
    entry.setdefault("tags", "")
    cur.execute("""
        INSERT INTO entries (date, period, type, category, source, amount, status, notes, is_manual, import_batch_id, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry["date"], entry["period"], entry["type"], entry["category"], entry["source"],
        float(entry["amount"]), entry["status"], entry["notes"], int(entry["is_manual"]),
        entry.get("import_batch_id"), entry["tags"], now_ts(), now_ts()
    ))
    conn.commit()
    conn.close()

def update_entry(entry_id: int, entry: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE entries SET date=?, period=?, type=?, category=?, source=?, amount=?, status=?, notes=?, is_manual=?, tags=?, updated_at=?
        WHERE id=?
    """, (
        entry["date"], entry["period"], entry["type"], entry["category"], entry["source"],
        float(entry["amount"]), entry["status"], entry["notes"], int(entry["is_manual"]),
        entry["tags"], now_ts(), entry_id
    ))
    conn.commit()
    conn.close()

def delete_entry(entry_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()

def load_entries_df(period: str | None = None) -> pd.DataFrame:
    conn = get_conn()
    if period:
        df = pd.read_sql_query("SELECT * FROM entries WHERE period=? ORDER BY date,id", conn, params=(period,))
    else:
        df = pd.read_sql_query("SELECT * FROM entries ORDER BY period,date,id", conn)
    conn.close()
    if not df.empty:
        df["amount"] = df["amount"].round(2)
    return df

def add_rule(pattern, category, type_, field, regex=False, priority=100):
    conn = get_conn()
    conn.execute("""
        INSERT INTO rules (pattern, category, type, field, regex, priority)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (pattern, category, type_, field, int(bool(regex)), int(priority)))
    conn.commit()
    conn.close()

def delete_rule(rule_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
    conn.commit()
    conn.close()

def load_rules():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM rules ORDER BY priority ASC, id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_template(name: str):
    conn = get_conn()
    conn.execute("INSERT INTO templates (name, created_at) VALUES (?,?)", (name, now_ts()))
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid() as id").fetchone()
    conn.close()
    return row["id"]

def list_templates():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM templates ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_template(template_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM template_items WHERE template_id=?", (template_id,))
    conn.execute("DELETE FROM templates WHERE id=?", (template_id,))
    conn.commit()
    conn.close()

def add_template_item(tid: int, item: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO template_items (template_id, type, category, source, amount, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (tid, item["type"], item["category"], item.get("source",""), float(item["amount"]), item.get("status","Planned"), item.get("notes","")))
    conn.commit()
    conn.close()

def get_template_items(tid: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM template_items WHERE template_id=? ORDER BY id", conn, params=(tid,))
    conn.close()
    return df

def apply_template_to_month(tid: int, period: str, set_status="Pending", mark_manual=True):
    items = get_template_items(tid)
    if items.empty:
        return 0
    year, month = map(int, period.split("-"))
    dt = date(year, month, 1)
    for _, r in items.iterrows():
        add_entry({
            "date": dt.isoformat(),
            "period": period,
            "type": r["type"],
            "category": r["category"],
            "source": r.get("source") or "",
            "amount": float(r["amount"]),
            "status": set_status,
            "notes": r.get("notes") or "",
            "is_manual": 1 if mark_manual else 0,
            "tags": ""
        })
    return len(items)

def set_budget(period: str, category: str, amount: float):
    conn = get_conn()
    # Upsert
    row = conn.execute("SELECT id FROM budgets WHERE period=? AND category=?", (period, category)).fetchone()
    if row:
        conn.execute("UPDATE budgets SET amount=? WHERE id=?", (float(amount), row["id"]))
    else:
        conn.execute("INSERT INTO budgets (period, category, amount) VALUES (?,?,?)", (period, category, float(amount)))
    conn.commit()
    conn.close()

def get_budgets(period: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM budgets WHERE period=?", conn, params=(period,))
    conn.close()
    return df

def add_import_batch(filename: str) -> int:
    conn = get_conn()
    conn.execute("INSERT INTO import_batches (filename, imported_at) VALUES (?,?)", (filename, now_ts()))
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid() as id").fetchone()
    conn.close()
    return row["id"]

# ------------- Categorization -------------
def categorize_with_rules(rules, type_, source, notes, category):
    text_fields = {
        "source": (source or "").lower(),
        "notes": (notes or "").lower(),
        "category": (category or "").lower(),
        "any": " ".join([source or "", notes or "", category or ""]).lower()
    }
    for r in rules:
        if r["type"] not in ("Any", type_):
            continue
        hay = text_fields.get(r["field"], text_fields["any"])
        pat = r["pattern"]
        if r["regex"]:
            try:
                if re.search(pat, hay, flags=re.IGNORECASE):
                    return r["category"]
            except re.error:
                pass
        else:
            if pat.lower() in hay:
                return r["category"]
    return None

KEYWORD_CATEGORY_MAP = [
    (r"rent", "Rent"),
    (r"car|fuel|gas|diesel|parking|uber|lyft", "Transportation"),
    (r"utility|electric|power|water|gas bill", "Utilities"),
    (r"phone|mobile|cell", "Phone"),
    (r"laya|insurance|insur", "Insurance"),
    (r"grocery|market|supermarket|walmart|costco|tesco|aldi|lidl", "Groceries"),
    (r"restaurant|cafe|coffee|starbucks|mcdonald|kfc|burger|pizza|bar|pub", "Dining"),
    (r"entertainment|cinema|movie|game|xbox|psn|steam", "Entertainment"),
    (r"travel|hotel|airbnb|airline|flight|booking|expedia", "Travel"),
    (r"education|course|udemy|coursera|school|tuition", "Education"),
    (r"clothes|apparel|zara|uniqlo|nike|adidas|h&m", "Shopping"),
    (r"electronics|apple|best buy|newegg|micro center", "Electronics"),
    (r"health|pharmacy|cvs|walgreens|clinic", "Healthcare"),
]

def categorize_with_heuristics(source, notes, category):
    text = " ".join([source or "", notes or "", category or ""]).lower()
    for pat, cat in KEYWORD_CATEGORY_MAP:
        if re.search(pat, text):
            return cat
    return "Uncategorized"

def match_import_to_manual(period, type_, amount):
    # Try match to manual entries by same type, same period, same amount (±0.01)
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM entries
        WHERE period=? AND type=? AND is_manual=1 AND ABS(amount - ?) < 0.01
        ORDER BY created_at DESC
        LIMIT 1
    """, (period, type_, float(amount))).fetchall()
    conn.close()
    if rows:
        r = dict(rows[0])
        return r.get("category") or None
    return None

def auto_categorize(row, rules):
    # Only auto-categorize expenses missing meaningful category
    if row["type"] != "Expense":
        return row.get("category") or ""
    category = (row.get("category") or "").strip()
    if category:
        return category
    # Try link to a manual entry
    mcat = match_import_to_manual(row["period"], row["type"], row["amount"])
    if mcat:
        return mcat
    # Try rules
    rcat = categorize_with_rules(rules, row["type"], row.get("source"), row.get("notes"), category)
    if rcat:
        return rcat
    # Try heuristics
    return categorize_with_heuristics(row.get("source"), row.get("notes"), category)

# ------------- Seed sample data -------------
def seed_example_data(year: int, skip_duplicates: bool = True, replace: bool = False) -> int:
    june = f"{year:04d}-06"
    jdate = date(year, 6, 1).isoformat()

    samples = [
        {"type":"Income","category":"Bank","source":"Bank","amount":23600,"status":"Paid"},
        {"type":"Income","category":"Cash/Saving","source":"Cash/Saving","amount":3000,"status":"Paid"},
        {"type":"Income","category":"Salary","source":"Salary","amount":6000,"status":"Paid"},
        {"type":"Income","category":"Expenses","source":"Expenses","amount":145,"status":"Paid"},
        {"type":"Income","category":"Additional","source":"Additional","amount":0,"status":"Paid"},
        {"type":"Expense","category":"Rent","source":"Rent","amount":1650,"status":"Paid"},
        {"type":"Expense","category":"Car","source":"Car","amount":360,"status":"Paid"},
        {"type":"Expense","category":"Utilities","source":"Utilities","amount":150,"status":"Pending"},
        {"type":"Expense","category":"Laya","source":"Laya","amount":150,"status":"Paid"},
        {"type":"Expense","category":"Phone","source":"Phone","amount":55,"status":"Pending"},
    ]

    if replace:
        conn = get_conn()
        conn.execute("DELETE FROM entries WHERE period=? AND notes='seed'", (june,))
        conn.commit()
        conn.close()

    existing_keys = set()
    if skip_duplicates:
        conn = get_conn()
        rows = conn.execute(
            "SELECT type, category, source, amount FROM entries WHERE period=? AND notes='seed'",
            (june,)
        ).fetchall()
        conn.close()
        existing_keys = {(r["type"], r["category"], r["source"], round(float(r["amount"]),2)) for r in rows}

    added = 0
    for s in samples:
        key = (s["type"], s["category"], s["source"], round(float(s["amount"]),2))
        if skip_duplicates and key in existing_keys:
            continue
        add_entry({
            "date": jdate,
            "period": june,
            "type": s["type"],
            "category": s["category"],
            "source": s["source"],
            "amount": float(s["amount"]),
            "status": s["status"],
            "notes": "seed",
            "is_manual": 1 if s["type"] == "Expense" else 0,
            "tags": ""
        })
        added += 1
    return added

# ------------- Insights -------------
def calc_month_summary(period: str):
    df = load_entries_df(period)
    if df.empty:
        return {
            "income": 0.0,
            "expense": 0.0,
            "net": 0.0,
            "by_cat": pd.DataFrame(columns=["category","amount"]),
            "df": df
        }
    income = df[df["type"]=="Income"]["amount"].sum()
    expense = df[df["type"]=="Expense"]["amount"].sum()
    by_cat = (df[df["type"]=="Expense"]
              .groupby("category", dropna=False)["amount"].sum()
              .sort_values(ascending=False)
              .reset_index())
    net = income - expense
    return {"income": income, "expense": expense, "net": net, "by_cat": by_cat, "df": df}

def trend_last_n_months(n=6):
    df = load_entries_df()
    if df.empty:
        return pd.DataFrame(columns=["period","Income","Expense","Net"])
    # Keep last n months by period
    periods = sorted(df["period"].unique())
    periods = periods[-n:]
    out = []
    for p in periods:
        d = df[df["period"]==p]
        inc = d[d["type"]=="Income"]["amount"].sum()
        exp = d[d["type"]=="Expense"]["amount"].sum()
        out.append({"period": p, "Income": inc, "Expense": exp, "Net": inc-exp})
    return pd.DataFrame(out)

def forecast_next_month():
    # naive forecast: average of last 3 months by category for expenses
    hist = trend_last_n_months(6)
    if hist.empty:
        return None
    df = load_entries_df()
    # compute last 3 periods
    periods = sorted(df["period"].unique())[-3:]
    if not periods:
        return None
    d3 = df[df["period"].isin(periods)]
    exp_by_cat = d3[d3["type"]=="Expense"].groupby("category")["amount"].mean().reset_index()
    return exp_by_cat

# ------------- UI -------------
st.set_page_config(page_title="Budget Planner", layout="wide")
st.title("Budget Planner")

init_db()

# Sidebar Month selector
with st.sidebar:
    st.header("Controls")
    periods = existing_periods()
    default_period = to_period(date.today())
    if default_period not in periods:
        periods.append(default_period)
        periods = sorted(periods)
    sel_period = st.selectbox("Select month", periods, index=periods.index(default_period))
    st.session_state["period"] = sel_period

    st.markdown("---")
    st.subheader("Sample data")
    st.caption("Adds 10 demo entries to June of the current year to preview charts.")

    skip_dups = st.checkbox("Skip duplicates",value=True,help="Avoid adding the same sample rows again.")
    replace_seed = st.checkbox("Replace existing",value=False,help="Delete existing June sample rows (notes='seed'), then add fresh ones.")

    if st.button("Add sample entries now", use_container_width=True):
        # Assumes you’re using the updated seed_example_data(year, skip_duplicates, replace)
        added = seed_example_data(year=date.today().year,skip_duplicates=skip_dups,replace=replace_seed)
        st.success(
            f"Replaced sample data. Added {added} entries."
            if replace_seed else
            f"Added {added} new sample entries."
        )
        st.rerun()

    st.markdown("---")
    st.caption("Tip: Deploy to Streamlit Cloud for a shareable link.")

tabs = st.tabs(["Dashboard","Entries","Import","Templates","Rules","Budgets","Settings"])

# Dashboard
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    summary = calc_month_summary(st.session_state["period"])
    income = summary["income"]
    expense = summary["expense"]
    net = summary["net"]
    col1.metric("Income", f"{income:,.2f}")
    col2.metric("Expenses", f"{expense:,.2f}")
    col3.metric("Net", f"{net:,.2f}")

    # Charts
    left, right = st.columns(2)
    by_cat = summary["by_cat"]
    if not by_cat.empty:
        fig_pie = px.pie(by_cat, names="category", values="amount", hole=0.5, title="Expense by Category")
        left.plotly_chart(fig_pie, use_container_width=True)
    trend = trend_last_n_months(6)
    if not trend.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend["period"], y=trend["Income"], name="Income"))
        fig_trend.add_trace(go.Bar(x=trend["period"], y=trend["Expense"], name="Expense"))
        fig_trend.add_trace(go.Scatter(x=trend["period"], y=trend["Net"], name="Net", mode="lines+markers"))
        fig_trend.update_layout(barmode="group", title="Last 6 Months")
        right.plotly_chart(fig_trend, use_container_width=True)

    # Budgets progress
    st.subheader(f"Budgets for {st.session_state['period']}")
    budgets = get_budgets(st.session_state["period"])
    if not budgets.empty:
        exp = summary["df"]
        exp = exp[(exp["type"]=="Expense") & (exp["period"]==st.session_state["period"])]
        spent_by_cat = exp.groupby("category")["amount"].sum()

        dfb = budgets.merge(spent_by_cat.rename("spent"), how="left", left_on="category", right_index=True)
        dfb["spent"] = dfb["spent"].fillna(0.0)
        dfb["within"] = np.minimum(dfb["spent"], dfb["amount"])
        dfb["over"] = np.maximum(0, dfb["spent"] - dfb["amount"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=dfb["category"], x=dfb["within"], orientation="h",
            name="Within budget", marker_color="#2ecc71"
        ))
        fig.add_trace(go.Bar(
            y=dfb["category"], x=dfb["over"], orientation="h",
            name="Over budget", marker_color="#e74c3c"
        ))
        fig.update_layout(
            barmode="stack",
            title=f"Budget vs Spent — {st.session_state['period']}",
            xaxis_title="Amount", yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        total_budget = float(dfb["amount"].sum())
        total_spent = float(dfb["spent"].sum())
        st.caption(f"Total: spent {total_spent:,.2f} of budget {total_budget:,.2f}")
    else:
        st.info("No budgets set for this month. Go to the Budgets tab to add them.")

    # Insights text
    if not summary["df"].empty:
        st.subheader("Insights")
        sr = 0.0 if income == 0 else (100.0 * (income - expense) / income)
        top_cats = by_cat.head(3)["category"].tolist() if not by_cat.empty else []
        bullets = [
            f"Savings rate: {sr:.1f}%",
            f"Top spending categories: {', '.join(top_cats) if top_cats else 'n/a'}"
        ]
        for b in bullets:
            st.write(f"- {b}")

        # Forecast
        fc = forecast_next_month()
        if fc is not None and not fc.empty:
            st.subheader("Next-month expense forecast (avg of last 3 months)")
            fig_fc = px.bar(fc, x="category", y="amount", title="Forecast by Category")
            st.plotly_chart(fig_fc, use_container_width=True)

# Entries
with tabs[1]:
    st.subheader(f"Entries for {st.session_state['period']}")
    df = load_entries_df(st.session_state["period"])
    if df.empty:
        st.info("No entries yet.")
    else:
        st.dataframe(df[["id","date","type","category","source","amount","status","is_manual","notes","tags"]], use_container_width=True, height=350)

    st.markdown("### Add new entry")
    with st.form("add_entry_form"):
        c1, c2, c3, c4 = st.columns(4)
        dt = c1.date_input("Date", value=date.today())
        etype = c2.selectbox("Type", ["Income","Expense"])
        amount = c3.number_input("Amount", step=1.0, format="%.2f")
        status = c4.selectbox("Status", ["Paid","Pending","Planned"])
        c5, c6, c7, c8 = st.columns(4)
        category = c5.text_input("Category")
        source = c6.text_input("Source/Payee")
        is_manual = c7.checkbox("Manual entry?", value=True)
        tags = c8.text_input("Tags (comma-separated)")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Add entry")
        if submitted:
            add_entry({
                "date": dt.isoformat(),
                "period": to_period(dt),
                "type": etype,
                "category": category.strip(),
                "source": source.strip(),
                "amount": amount,
                "status": status,
                "notes": notes.strip(),
                "is_manual": 1 if is_manual else 0,
                "tags": tags.strip()
            })
            st.success("Entry added. Reloading...")
            st.rerun()

    st.markdown("### Edit / Delete")
    df = load_entries_df(st.session_state["period"])
    if not df.empty:
        options = df.apply(lambda r: f"#{r['id']} | {r['date']} | {r['type']} | {r['category']} | {r['amount']}", axis=1).tolist()
        id_map = {opt: int(opt.split("|")[0].strip().replace("#","")) for opt in options}
        sel = st.selectbox("Select entry to edit", [""] + options)
        if sel:
            eid = id_map[sel]
            row = df[df["id"]==eid].iloc[0]
            with st.form("edit_entry_form"):
                c1, c2, c3, c4 = st.columns(4)
                dt = c1.date_input("Date", value=date.fromisoformat(row["date"]))
                etype = c2.selectbox("Type", ["Income","Expense"], index=0 if row["type"]=="Income" else 1)
                amount = c3.number_input("Amount", step=1.0, value=float(row["amount"]), format="%.2f")
                status = c4.selectbox("Status", ["Paid","Pending","Planned"], index=["Paid","Pending","Planned"].index(row["status"] if row["status"] in ["Paid","Pending","Planned"] else "Pending"))
                c5, c6, c7, c8 = st.columns(4)
                category = c5.text_input("Category", value=row["category"] or "")
                source = c6.text_input("Source/Payee", value=row["source"] or "")
                is_manual = c7.checkbox("Manual entry?", value=bool(row["is_manual"]))
                tags = c8.text_input("Tags (comma-separated)", value=row["tags"] or "")
                notes = st.text_area("Notes", value=row["notes"] or "", height=80)
                colA, colB = st.columns([1,1])
                save = colA.form_submit_button("Save changes")
                delbtn = colB.form_submit_button("Delete", type="secondary")
                if save:
                    update_entry(eid, {
                        "date": dt.isoformat(),
                        "period": to_period(dt),
                        "type": etype,
                        "category": category.strip(),
                        "source": source.strip(),
                        "amount": amount,
                        "status": status,
                        "notes": notes.strip(),
                        "is_manual": 1 if is_manual else 0,
                        "tags": tags.strip()
                    })
                    st.success("Updated.")
                    st.rerun()
                if delbtn:
                    delete_entry(eid)
                    st.warning("Deleted.")
                    st.rerun()

# Import
with tabs[2]:
    st.subheader("Upload Excel/CSV")
    up = st.file_uploader("Upload file", type=["xlsx","xls","csv"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            raw = pd.read_csv(up)
        else:
            raw = pd.read_excel(up)
        st.write("Preview")
        st.dataframe(raw.head(20), use_container_width=True)

        st.info("Map columns. Use the best available columns from your file.")
        cols = list(raw.columns)
        c1, c2, c3 = st.columns(3)
        col_date = c1.selectbox("Date column (or None)", ["<None>"]+cols)
        col_month = c2.selectbox("Month name column (or None)", ["<None>"]+cols, index=cols.index("Month") if "Month" in cols else 0)
        col_type = c3.selectbox("Type column", cols if "Type" in cols else ["<None>"]+cols, index=cols.index("Type") if "Type" in cols else 0)
        c4, c5, c6 = st.columns(3)
        col_cat = c4.selectbox("Source/Category column", cols if "Source/Category" in cols else ["<None>"]+cols, index=cols.index("Source/Category") if "Source/Category" in cols else 0)
        col_amount = c5.selectbox("Amount column", cols if "Amount" in cols else ["<None>"]+cols, index=cols.index("Amount") if "Amount" in cols else 0)
        col_status = c6.selectbox("Status column", ["<None>"]+cols, index=cols.index("Status") if "Status" in cols else 0)
        col_desc = st.selectbox("Description/Notes column (optional)", ["<None>"]+cols)

        if col_month != "<None>":
            year_for_month = st.number_input("Year for the 'Month' column", min_value=2000, max_value=2100, value=date.today().year)
        else:
            year_for_month = date.today().year

        auto_mark_manual = st.checkbox("Mark imported as manual?", value=False)
        do_autocat = st.checkbox("Auto-categorize missing expense categories", value=True)

        if st.button("Process and Import"):
            # Build normalized rows
            rules = load_rules()
            processed = []
            for _, r in raw.iterrows():
                type_ = (str(r[col_type]).strip() if col_type != "<None>" else "").title()
                if type_ not in ("Income","Expense"):
                    # attempt to infer: negative as expense, positive as income
                    try:
                        amt = float(r[col_amount])
                        type_ = "Expense" if amt < 0 else "Income"
                    except Exception:
                        type_ = "Expense"
                amount = float(r[col_amount])
                status = str(r[col_status]).title() if (col_status != "<None>" and not pd.isna(r[col_status])) else "Paid"
                source_cat = str(r[col_cat]) if col_cat != "<None>" else ""
                notes = str(r[col_desc]) if (col_desc != "<None>" and not pd.isna(r[col_desc])) else ""

                # Date/Period
                if col_date != "<None>" and not pd.isna(r[col_date]):
                    try:
                        dt = pd.to_datetime(r[col_date]).date()
                    except Exception:
                        dt = date(year_for_month, 1, 1)
                    period = to_period(dt)
                elif col_month != "<None>" and not pd.isna(r[col_month]):
                    m = month_name_to_num(str(r[col_month]))
                    m = m if m >= 1 else date.today().month
                    dt = date(year_for_month, m, 1)
                    period = ensure_period_from_parts(year_for_month, m)
                else:
                    dt = date.today()
                    period = to_period(dt)

                category = source_cat.strip()
                source = source_cat.strip()
                row = {
                    "date": dt.isoformat(),
                    "period": period,
                    "type": type_,
                    "category": category,
                    "source": source,
                    "amount": abs(amount),
                    "status": status,
                    "notes": notes,
                    "is_manual": 1 if auto_mark_manual else 0,
                    "tags": ""
                }
                if do_autocat and type_ == "Expense" and (not category or category.lower() in ("", "nan", "none", "uncategorized")):
                    row["category"] = auto_categorize(row, rules)
                processed.append(row)

            if not processed:
                st.warning("No rows processed.")
            else:
                batch_id = add_import_batch(up.name)
                for row in processed:
                    row["import_batch_id"] = batch_id
                    add_entry(row)
                st.success(f"Imported {len(processed)} rows into {len(set([r['period'] for r in processed]))} month(s).")
                st.rerun()

# Templates
with tabs[3]:
    st.subheader("Templates")
    templates = list_templates()
    tnames = ["<New template>"] + [f"{t['id']}: {t['name']}" for t in templates]
    sel = st.selectbox("Choose template", tnames)
    if sel == "<New template>":
        new_name = st.text_input("New template name")
        if st.button("Create template") and new_name.strip():
            tid = add_template(new_name.strip())
            st.success(f"Created template '{new_name}'.")
            st.rerun()
    else:
        tid = int(sel.split(":")[0])
        st.markdown("#### Template items")
        items = get_template_items(tid)
        if items.empty:
            st.info("No items yet.")
        else:
            st.dataframe(items[["id","type","category","source","amount","status","notes"]], use_container_width=True)
        st.markdown("Add item")
        with st.form("add_template_item"):
            c1,c2,c3,c4 = st.columns(4)
            ttype = c1.selectbox("Type", ["Expense","Income"])
            tcat = c2.text_input("Category/Source")
            tamount = c3.number_input("Amount", step=1.0, format="%.2f")
            tstatus = c4.selectbox("Status", ["Planned","Pending","Paid"])
            tnotes = st.text_input("Notes", "")
            addit = st.form_submit_button("Add item")
            if addit:
                add_template_item(tid, {"type": ttype, "category": tcat, "source": tcat, "amount": tamount, "status": tstatus, "notes": tnotes})
                st.success("Item added.")
                st.rerun()

        st.markdown("#### Apply template to month")
        ap_period = st.selectbox("Month to apply", existing_periods() + [st.session_state["period"]], index=existing_periods().index(st.session_state["period"]) if st.session_state["period"] in existing_periods() else 0)
        ap_status = st.selectbox("Set status", ["Pending","Planned","Paid"])
        ap_manual = st.checkbox("Mark as manual?", value=True)
        if st.button("Apply template"):
            count = apply_template_to_month(tid, ap_period, set_status=ap_status, mark_manual=ap_manual)
            st.success(f"Applied {count} items to {ap_period}.")
            st.rerun()

        st.markdown("---")
        if st.button("Delete template", type="secondary"):
            delete_template(tid)
            st.warning("Template deleted.")
            st.rerun()

    st.markdown("#### Save current month as template")
    new_tname = st.text_input("Template name", value=f"{st.session_state['period']} plan")
    if st.button("Save month entries as template"):
        dfm = load_entries_df(st.session_state["period"])
        if dfm.empty:
            st.info("No entries to save.")
        else:
            new_tid = add_template(new_tname.strip() or f"Template {now_ts()}")
            for _, r in dfm.iterrows():
                add_template_item(new_tid, {"type": r["type"], "category": r["category"], "source": r["source"], "amount": r["amount"], "status": r["status"], "notes": r.get("notes") or ""})
            st.success(f"Saved {len(dfm)} items to template '{new_tname}'.")
            st.rerun()

# Rules
with tabs[4]:
    st.subheader("Auto-categorization rules")
    rules = load_rules()
    if rules:
        rdf = pd.DataFrame(rules)
        st.dataframe(rdf[["id","pattern","category","type","field","regex","priority"]], use_container_width=True)
    else:
        st.info("No rules yet. Add one below.")

    with st.form("add_rule_form"):
        c1,c2,c3 = st.columns(3)
        pattern = c1.text_input("Pattern (substring or regex)")
        rcat = c2.text_input("Category to assign")
        rtype = c3.selectbox("Type match", ["Any","Expense","Income"], index=0)
        c4,c5,c6 = st.columns(3)
        field = c4.selectbox("Field to search", ["any","source","notes","category"], index=0)
        regex = c5.checkbox("Use regex?", value=False)
        priority = c6.number_input("Priority (lower = earlier)", min_value=1, max_value=999, value=100)
        addbtn = st.form_submit_button("Add rule")
        if addbtn and pattern.strip() and rcat.strip():
            add_rule(pattern.strip(), rcat.strip(), rtype, field, regex=regex, priority=priority)
            st.success("Rule added.")
            st.rerun()

    del_id = st.number_input("Delete rule by ID", min_value=0, step=1)
    if st.button("Delete rule"):
        if del_id > 0:
            delete_rule(int(del_id))
            st.warning(f"Rule {int(del_id)} deleted.")
            st.rerun()

# Budgets
with tabs[5]:
    st.subheader(f"Budgets for {st.session_state['period']}")
    # Suggest categories from current month expenses
    dfm = load_entries_df(st.session_state["period"])
    cats = sorted(set(dfm[dfm["type"]=="Expense"]["category"])) if not dfm.empty else []
    if not cats:
        st.info("No expense categories found for this month. You can still add budgets manually below.")
    c1, c2 = st.columns([2,1])
    new_cat = c1.text_input("Category")
    new_amt = c2.number_input("Budget amount", step=10.0, format="%.2f")
    if st.button("Set/Update budget"):
        if new_cat.strip():
            set_budget(st.session_state["period"], new_cat.strip(), new_amt)
            st.success("Budget updated.")
            st.rerun()

    budgets = get_budgets(st.session_state["period"])
    if not budgets.empty:
        st.dataframe(budgets, use_container_width=True)

# Settings
with tabs[6]:
    st.subheader("Export (selected month)")
    df_month = load_entries_df(st.session_state["period"])
    if not df_month.empty:
        csv_month = df_month.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV for selected month",
            csv_month,
            file_name=f"entries_{st.session_state['period']}.csv",
            mime="text/csv"
        )
    else:
        st.info("No entries to export for this month.")

    st.markdown("### Export all data")
    df_all = load_entries_df()
    if not df_all.empty:
        csv_all = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full CSV (all months)",
            csv_all,
            file_name="entries_all.csv",
            mime="text/csv"
        )
    else:
        st.caption("No data available yet.")

    # Optional: quick DB backup download
    st.markdown("### Database backup")
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            st.download_button(
                "Download SQLite database (budget.db)",
                f,
                file_name="budget.db",
                mime="application/octet-stream"
            )
    else:
        st.caption("Database file not found (it will be created automatically when you add data).")

    st.markdown("---")
    with st.expander("Manage sample data (June)"):
        st.caption("These are the demo rows inserted via the 'Add sample entries' action and labeled with notes='seed'.")
        c1, c2 = st.columns(2)
        yr = c1.number_input("Year for June sample", min_value=2000, max_value=2100, value=date.today().year, step=1)
        remove_all_years = c2.checkbox("Target ALL years", value=False, help="If on, removes all seeded June entries across all years.")

        colA, colB, colC = st.columns(3)

        if colA.button("Remove sample (June)"):
            conn = get_conn()
            if remove_all_years:
                # Remove all seeded entries regardless of year
                conn.execute("DELETE FROM entries WHERE notes='seed' AND strftime('%m', date)='06'")
                msg = "Removed sample entries for June across all years."
            else:
                june_period = f"{int(yr):04d}-06"
                conn.execute("DELETE FROM entries WHERE period=? AND notes='seed'", (june_period,))
                msg = f"Removed sample entries for {june_period}."
            conn.commit()
            conn.close()
            st.warning(msg)
            st.rerun()

        if colB.button("Replace sample (June)"):
            # Delete then re-add fresh
            conn = get_conn()
            if remove_all_years:
                conn.execute("DELETE FROM entries WHERE notes='seed' AND strftime('%m', date)='06'")
            else:
                june_period = f"{int(yr):04d}-06"
                conn.execute("DELETE FROM entries WHERE period=? AND notes='seed'", (june_period,))
            conn.commit()
            conn.close()

            # Re-seed (supports both the new and old function signatures)
            added = None
            try:
                added = seed_example_data(year=int(yr), skip_duplicates=False, replace=False)
            except TypeError:
                try:
                    seed_example_data(int(yr))
                except Exception:
                    pass

            if added is not None:
                st.success(f"Re-seeded June {int(yr)} with {added} sample entries.")
            else:
                st.success(f"Re-seeded June {int(yr)}.")
            st.rerun()

        if colC.button("Remove ALL sample entries (any month/year)"):
            conn = get_conn()
            conn.execute("DELETE FROM entries WHERE notes='seed'")
            conn.commit()
            conn.close()
            st.warning("Removed all sample entries across all months and years.")
            st.rerun()

    st.markdown("---")
    # Danger zone
    if st.checkbox("Danger zone: show destructive actions"):
        st.caption("These actions cannot be undone.")
        if st.button("Wipe ALL data (delete local database)", type="secondary"):
            if DB_PATH.exists():
                DB_PATH.unlink()
            init_db()
            st.warning("Database wiped.")
            st.rerun()



