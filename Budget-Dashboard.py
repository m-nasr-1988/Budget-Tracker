import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
from pathlib import Path
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numbers
import numpy as np

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
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT CHECK(type IN ('Income','Expense')),
        name TEXT,
        is_default INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(type, name)
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

def seed_default_categories():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
    if n == 0:
        income = [
            "Salary","Cash/Saving","Bonus",
            "Investment Income","Refunds","Additional","Other"
        ]
        expenses = [
            "Rent","Utilities","Fuel","Phone","Internet","Water","Electricity","Gas Bill",
            "Health Insurance","Insurance",
            "Groceries","Dining","Transportation","Car","Parking",
            "Healthcare","Medication",
            "Subscriptions","Entertainment","Education","Electronics",
            "Travel","Shopping","Other"
        ]
        for name in income:
            conn.execute("INSERT OR IGNORE INTO categories (type,name,is_default) VALUES ('Income', ?, 1)", (name,))
        for name in expenses:
            conn.execute("INSERT OR IGNORE INTO categories (type,name,is_default) VALUES ('Expense', ?, 1)", (name,))
        conn.commit()
    conn.close()

def get_categories(type_: str) -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT name FROM categories WHERE type=? ORDER BY name", (type_,)).fetchall()
    conn.close()
    lst = [r["name"] for r in rows]
    if "Other" not in lst and "Others" not in lst:
        lst.append("Other")
    return lst

def get_all_categories() -> list[str]:
    # For the inline table editor we use a unified list
    return sorted(set(get_categories("Income") + get_categories("Expense")))

def add_category(type_: str, name: str) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    conn = get_conn()
    try:
        conn.execute("INSERT OR IGNORE INTO categories (type,name,is_default) VALUES (?,?,0)", (type_, name))
        conn.commit()
        return True
    finally:
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

def parse_amount_text(s) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # Keep digits, minus, dot, comma; drop other chars
    import re as _re
    cleaned = _re.sub(r"[^\d\-\.,]", "", s)
    # Normalize: remove thousands separators
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except Exception:
        return None

def safe_clear_state(key):
    if key not in st.session_state:
        return
    v = st.session_state[key]
    try:
        if isinstance(v, bool):
            st.session_state[key] = False
        elif isinstance(v, (numbers.Integral, numbers.Real, np.number)):
            st.session_state[key] = 0.0
        elif isinstance(v, (list, tuple, set)):
            st.session_state[key] = type(v)()  # empty same type
        else:
            st.session_state[key] = ""
    except Exception:
        # If Streamlit refuses to set (rare), just drop the key
        st.session_state.pop(key, None)

def guess_column(cols, patterns):
    # Best-effort guess: exact match > startswith > contains
    cols_l = [c.lower().strip() for c in cols]
    score = {i: 0 for i in range(len(cols))}
    for i, cl in enumerate(cols_l):
        for rank, pat in enumerate(patterns):
            p = pat.lower()
            if cl == p:
                score[i] += 100 - rank  # exact
            elif cl.startswith(p):
                score[i] += 50 - rank   # startswith
            elif p in cl:
                score[i] += 20 - rank   # contains
    best_i = max(score, key=score.get) if cols else None
    return cols[best_i] if best_i is not None and score[best_i] > 0 else None

def select_index_with_guess(all_options, guess):
    # all_options includes "<None>" + cols
    if not guess:
        return 0
    try:
        return all_options.index(guess)
    except ValueError:
        return 0


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


def render_settings_sidebar():
    st.markdown("Export")
    # Selected month
    df_month = load_entries_df(st.session_state["period"])
    if not df_month.empty:
        csv_month = df_month.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Month CSV",
            csv_month,
            file_name=f"entries_{st.session_state['period']}.csv",
            mime="text/csv",
            key="sb_dl_month_csv",
            use_container_width=True,
        )
    else:
        st.caption("No entries to export for this month.")

    # All data
    df_all = load_entries_df()
    if not df_all.empty:
        csv_all = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ All data CSV",
            csv_all,
            file_name="entries_all.csv",
            mime="text/csv",
            key="sb_dl_all_csv",
            use_container_width=True,
        )
    else:
        st.caption("No data available yet.")

    # SQLite backup download
    st.markdown("Database")
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download budget.db",
                f,
                file_name="budget.db",
                mime="application/octet-stream",
                key="sb_dl_db",
                use_container_width=True,
            )
    else:
        st.caption("Database file will be created automatically when you add data.")

    st.markdown("---")
    st.markdown("Sample data")
    st.caption("Manage demo rows inserted via the sample seeding (notes='seed').")

    yr = st.number_input(
        "Year for June sample",
        min_value=2000,
        max_value=2100,
        value=date.today().year,
        step=1,
        key="sb_seed_year",
    )

    if st.button("Remove June sample (selected year)", key="sb_btn_rm_june_seed", use_container_width=True):
        conn = get_conn()
        june_period = f"{int(yr):04d}-06"
        conn.execute("DELETE FROM entries WHERE period=? AND notes='seed'", (june_period,))
        conn.commit()
        conn.close()
        st.warning(f"Removed sample entries for {june_period}.")
        st.rerun()

    if st.button("Replace June sample (selected year)", key="sb_btn_replace_june_seed", use_container_width=True):
        # Remove then re-seed
        conn = get_conn()
        june_period = f"{int(yr):04d}-06"
        conn.execute("DELETE FROM entries WHERE period=? AND notes='seed'", (june_period,))
        conn.commit()
        conn.close()
        # Re-seed with backward-compatible call
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

    if st.button("Remove ALL sample entries (any month/year)", key="sb_btn_rm_all_seed", use_container_width=True):
        conn = get_conn()
        conn.execute("DELETE FROM entries WHERE notes='seed'")
        conn.commit()
        conn.close()
        st.warning("Removed all sample entries.")
        st.rerun()

    st.markdown("---")
    with st.expander("Danger zone", expanded=False):
        st.caption("Irreversible actions.")
        if st.button("Wipe ALL data (delete local database)", type="secondary", key="sb_btn_wipe_all", use_container_width=True):
            if DB_PATH.exists():
                DB_PATH.unlink()
            init_db()
            st.warning("Database wiped.")
            st.rerun()

# ------------- UI -------------
st.set_page_config(page_title="Budget Planner", layout="wide")
st.title("Budget Planner")

init_db()
seed_default_categories()  # ensure default category lists exist
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

    # Compact toolbar row of icon popovers
    st.markdown("---")
    c1, c2 = st.columns([1, 1], gap="small")
    
    # ⚙️ Settings (Export, DB, Seed mgmt, Danger zone)
    with c1:
        if hasattr(st, "popover"):
            with st.popover("⚙️"):
                render_settings_sidebar()  # reuse your existing function
        else:
            with st.expander("⚙️ Settings", expanded=False):
                render_settings_sidebar()
    
    # 🧪 Sample data (optional separate icon; remove this block if you keep it inside ⚙️)
    def render_sample_data_quick():
        st.caption("Add demo rows to June (current year)")
        skip_dups = st.checkbox("Skip duplicates", value=True, key="sb_sd_skip")
        replace_seed = st.checkbox("Replace existing", value=False, key="sb_sd_replace")
        if st.button("Add sample entries", key="sb_sd_add", use_container_width=True):
            try:
                added = seed_example_data(year=date.today().year, skip_duplicates=skip_dups, replace=replace_seed)
            except TypeError:
                # Backward compatibility with older seed function
                added = None
                seed_example_data(date.today().year)
            st.success(f"Added {added or 10} entries.")  # 10 in the default seed
            st.rerun()
    
    with c2:
        if hasattr(st, "popover"):
            with st.popover("🧪"):
                render_sample_data_quick()
        else:
            with st.expander("🧪 Sample data", expanded=False):
                render_sample_data_quick()

    st.markdown("---")
    st.caption("Tip: Deploy to Streamlit Cloud for a shareable link.")

show_rules_tab = False  # toggle to True when you want to show the Rules tab

tab_names = ["Dashboard", "Entries", "Import", "Templates"]
if show_rules_tab:
    tab_names.append("Rules")
tab_names += ["Budgets"]

tab_objs = st.tabs(tab_names)
tab = {name: tab_objs[i] for i, name in enumerate(tab_names)}

# Dashboard
with tab["Dashboard"]:
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
        # Build a clean Month/Year label from period
        trend = trend.copy()
        try:
            # If period is like "2025-05", turn it into "May 2025"
            trend["label"] = pd.to_datetime(trend["period"] + "-01").dt.strftime("%b %Y")
        except Exception:
            # Fallback: just use the raw value
            trend["label"] = trend["period"].astype(str)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend["label"], y=trend["Income"], name="Income"))
        fig_trend.add_trace(go.Bar(x=trend["label"], y=trend["Expense"], name="Expense"))
        fig_trend.add_trace(go.Scatter(x=trend["label"], y=trend["Net"], name="Net", mode="lines+markers"))
        fig_trend.update_layout(barmode="group", title="Last 6 Months")

        # Treat x as categories so Plotly doesn’t auto-convert to datetime (no time displayed)
        fig_trend.update_xaxes(type="category")

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
with tab["Entries"]:
    st.subheader(f"Entries for {st.session_state['period']}")

    # Search/filter
    q = st.text_input("Search", placeholder="Filter by text, category, source, status, notes, tags…")
    df_month = load_entries_df(st.session_state["period"])
    df_show = df_month.copy()
    if not df_show.empty and q:
        ql = q.lower().strip()
        cols_to_search = ["date","type","category","source","status","notes","tags"]
        df_show = df_show[df_show.apply(
            lambda r: ql in " ".join([str(r.get(c, "") or "") for c in cols_to_search]).lower(),
            axis=1
        )]

    # --- Quick inline edit ---
    st.markdown("### Quick inline edit")
    if df_show.empty:
        st.info("No entries to show for this month.")
    else:
        # Preserve is_manual for this month (since we hid it from the grid)
        manual_map = dict(zip(df_month["id"].astype(int), df_month["is_manual"].astype(int)))
    
        # Columns to show/edit (is_manual removed)
        edit_cols = ["id", "date", "type", "category", "source", "amount", "status", "notes", "tags"]
        view = df_show[edit_cols].copy()
        view["date"] = pd.to_datetime(view["date"], errors="coerce")
        view["amount"] = pd.to_numeric(view["amount"], errors="coerce").astype(float)
    
        edited = st.data_editor(
            view,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            disabled=["id"],
            column_config={
                "id": st.column_config.TextColumn("ID"),
                "date": st.column_config.DateColumn("Date"),
                "type": st.column_config.SelectboxColumn("Type", options=["Income", "Expense"]),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=sorted(set(get_categories("Income") + get_categories("Expense")))
                ),
                "source": st.column_config.TextColumn("Source/Payee"),
                "amount": st.column_config.NumberColumn("Amount", step=0.01, format="%.2f", min_value=0.0),
                "status": st.column_config.SelectboxColumn("Status", options=["Paid", "Pending", "Planned"]),
                "notes": st.column_config.TextColumn("Notes"),
                "tags": st.column_config.TextColumn("Tags"),
            },
            key=f"editor_{st.session_state['period']}"
        )
    
        if st.button("Save table edits", type="primary"):
            changed = 0
            for _, r in edited.iterrows():
                try:
                    eid = int(r["id"])
                except Exception:
                    continue
    
                # Keep the original manual flag
                old_manual = int(manual_map.get(eid, 0))
    
                # Normalize date
                dt = r["date"]
                if pd.isna(dt):
                    dt = date.today()
                elif isinstance(dt, pd.Timestamp):
                    dt = dt.date()
    
                update_entry(eid, {
                    "date": dt.isoformat(),
                    "period": to_period(dt),
                    "type": str(r["type"]),
                    "category": str(r["category"] or "").strip(),
                    "source": str(r["source"] or "").strip(),
                    "amount": float(r["amount"] or 0.0),
                    "status": str(r["status"] or "Pending"),
                    "notes": str(r["notes"] or "").strip(),
                    "is_manual": old_manual,  # preserved
                    "tags": str(r["tags"] or "").strip(),
                })
                changed += 1
    
            st.success(f"Saved edits to {changed} row(s).")
            st.rerun()
    
        st.markdown("---")
        st.markdown("### Add new entry (dynamic)")
    
        # Reset category when Type changes
        def _reset_add_category():
            for k in ("add_category", "add_new_category_name"):
                if k in st.session_state:
                    del st.session_state[k]
    
        c1, c2, c3, c4 = st.columns(4)
        dt = c1.date_input("Date", value=date.today(), key="add_date")
        etype = c2.selectbox("Type", ["Income","Expense"], key="add_type", on_change=_reset_add_category)
        amount_text = c3.text_input("Amount", key="add_amount_text", placeholder="e.g. 1234.56")
        status = c4.selectbox("Status", ["Paid","Pending","Planned"], key="add_status")
    
        c5, c6, c7 = st.columns([2,2,2])
        cat_options = get_categories(etype)
        cat_options_plus = cat_options + ["+ Add new category..."]
        sel_cat = c5.selectbox(
            "Category",
            options=cat_options_plus,
            key="add_category",
            index=(cat_options.index(st.session_state["add_category"]) if "add_category" in st.session_state and st.session_state["add_category"] in cat_options else 0)
        )
        source = c6.text_input("Source/Payee", key="add_source")
        tags = c7.text_input("Tags", key="add_tags")
    
        if sel_cat == "+ Add new category...":
            new_cat_inline = st.text_input("New category name (for the selected Type)", key="add_new_category_name")
            if st.button("Add category to list", key="btn_add_cat_inline"):
                if new_cat_inline and new_cat_inline.strip():
                    if add_category(etype, new_cat_inline.strip()):
                        st.session_state["add_category"] = new_cat_inline.strip()
                        st.success(f"Added '{new_cat_inline.strip()}' under {etype}.")
                        st.rerun()
                    else:
                        st.warning("That category already exists or name is invalid.")
                else:
                    st.error("Please enter a category name.")
    
        notes = st.text_area("Notes", height=80, key="add_notes")
    
        if st.button("Add entry", type="primary", key="btn_add_entry"):
            amt = parse_amount_text(amount_text)
            if amt is None or amt < 0:
                st.error("Please enter a valid non-negative amount.")
            else:
                final_category = st.session_state.get("add_category")
                if final_category == "+ Add new category...":
                    new_name = st.session_state.get("add_new_category_name", "")
                    if not new_name or not new_name.strip():
                        st.error("Please add a new category name or choose an existing one.")
                        st.stop()
                    add_category(etype, new_name.strip())
                    final_category = new_name.strip()
    
                add_entry({
                    "date": dt.isoformat(),
                    "period": to_period(dt),
                    "type": etype,
                    "category": final_category,
                    "source": (source or "").strip(),
                    "amount": float(amt),
                    "status": status,
                    "notes": (notes or "").strip(),
                    "is_manual": 1,  # set automatically: UI-added entries are manual
                    "tags": (tags or "").strip()
                })
                st.success("Entry added.")
                # Clear text fields safely
                for k in ("add_source","add_tags","add_notes","add_amount_text"):
                    safe_clear_state(k)
                st.rerun()

        # --- Quick budgets (this month) ---
        with st.expander("Quick budgets (this month)"):
            exp_cats = get_categories("Expense")
            cat_plus = exp_cats + ["+ Add new category..."]
        
            c1, c2 = st.columns([2, 1.2])
            sel_budget_cat_e = c1.selectbox("Category", cat_plus, key="e_budget_cat_select")
            budget_amt_text_e = c2.text_input("Amount", value="", placeholder="e.g. 500.00", key="e_budget_amount_text")
        
            # Inline add new category
            if sel_budget_cat_e == "+ Add new category...":
                new_budget_cat_e = st.text_input("New expense category name", key="e_budget_new_cat_name")
                if st.button("Add category", key="e_btn_add_budget_cat"):
                    if new_budget_cat_e and new_budget_cat_e.strip():
                        if add_category("Expense", new_budget_cat_e.strip()):
                            st.session_state["e_budget_cat_select"] = new_budget_cat_e.strip()
                            st.success(f"Added '{new_budget_cat_e.strip()}' to Expense categories.")
                            st.rerun()
                        else:
                            st.warning("That category already exists or name is invalid.")
                    else:
                        st.error("Please enter a category name.")
        
            if st.button("Set/Update budget", key="e_btn_set_budget"):
                # Resolve category (handles "+ Add new...")
                if st.session_state["e_budget_cat_select"] == "+ Add new category...":
                    new_cat_name = st.session_state.get("e_budget_new_cat_name", "")
                    if not new_cat_name or not new_cat_name.strip():
                        st.error("Please add a new category name or choose an existing one.")
                        st.stop()
                    add_category("Expense", new_cat_name.strip())
                    final_cat_e = new_cat_name.strip()
                else:
                    final_cat_e = st.session_state["e_budget_cat_select"]
        
                # Parse amount (free-typed)
                amt_val_e = parse_amount_text(budget_amt_text_e)
                if amt_val_e is None or amt_val_e < 0:
                    st.error("Please enter a valid non-negative amount.")
                else:
                    set_budget(st.session_state["period"], final_cat_e, float(amt_val_e))
                    st.success(f"Budget set for {final_cat_e}: {amt_val_e:,.2f}")
                    # Clear amount field safely
                    st.session_state.pop("e_budget_amount_text", None)
                    st.rerun()
        
            st.markdown("Current budgets for this month")
            budgets_df_e = get_budgets(st.session_state["period"])
            if budgets_df_e.empty:
                st.info("No budgets set for this month yet.")
            else:
                st.dataframe(budgets_df_e, use_container_width=True)

        st.markdown("---")
        st.markdown("### Advanced: Edit or Delete a specific entry")
        df_adv = load_entries_df(st.session_state["period"])
        if df_adv.empty:
            st.info("No entries to edit.")
        else:
            options = df_adv.apply(lambda r: f"#{r['id']} | {r['date']} | {r['type']} | {r['category']} | {r['amount']}", axis=1).tolist()
            id_map = {opt: int(opt.split("|")[0].strip().replace("#","")) for opt in options}
            sel = st.selectbox("Select entry", [""] + options, key="edit_select")
            if sel:
                eid = id_map[sel]
                row = df_adv[df_adv["id"]==eid].iloc[0]
    
                c1, c2, c3, c4 = st.columns(4)
                edate = c1.date_input("Date", value=date.fromisoformat(row["date"]), key=f"edit_date_{eid}")
                etype2 = c2.selectbox("Type", ["Income","Expense"], index=0 if row["type"]=="Income" else 1, key=f"edit_type_{eid}")
                eamount_text = c3.text_input("Amount", value=str(row["amount"]), key=f"edit_amount_text_{eid}")
                estatus = c4.selectbox("Status", ["Paid","Pending","Planned"], index=["Paid","Pending","Planned"].index(row["status"] if row["status"] in ["Paid","Pending","Planned"] else "Pending"), key=f"edit_status_{eid}")
    
                c5, c6, c7 = st.columns([2,2,2])
                cat_opts2 = get_categories(etype2) + ["+ Add new category..."]
                current_cat = row["category"] if row["category"] in cat_opts2 else cat_opts2[0]
                ecat = c5.selectbox("Category", cat_opts2, index=cat_opts2.index(current_cat), key=f"edit_category_{eid}")
                esource = c6.text_input("Source/Payee", value=row["source"] or "", key=f"edit_source_{eid}")
                etags = c7.text_input("Tags", value=row["tags"] or "", key=f"edit_tags_{eid}")
                enotes = st.text_area("Notes", value=row["notes"] or "", height=80, key=f"edit_notes_{eid}")
    
                if ecat == "+ Add new category...":
                    new_cat_inline2 = st.text_input("New category name (for the selected Type)", key=f"edit_new_cat_name_{eid}")
                    if st.button("Add category to list", key=f"btn_edit_add_cat_{eid}"):
                        if new_cat_inline2 and new_cat_inline2.strip():
                            if add_category(etype2, new_cat_inline2.strip()):
                                st.session_state[f"edit_category_{eid}"] = new_cat_inline2.strip()
                                st.success(f"Added '{new_cat_inline2.strip()}' under {etype2}.")
                                st.rerun()
                            else:
                                st.warning("That category already exists or name is invalid.")
                        else:
                            st.error("Please enter a category name.")
    
                colA, colB = st.columns(2)
                if colA.button("Save changes", key=f"btn_save_{eid}"):
                    amt2 = parse_amount_text(eamount_text)
                    if amt2 is None or amt2 < 0:
                        st.error("Please enter a valid non-negative amount.")
                    else:
                        final_cat2 = st.session_state.get(f"edit_category_{eid}", ecat)
                        if final_cat2 == "+ Add new category...":
                            new_name2 = st.session_state.get(f"edit_new_cat_name_{eid}", "")
                            if not new_name2 or not new_name2.strip():
                                st.error("Please add a new category name or choose an existing one.")
                                st.stop()
                            add_category(etype2, new_name2.strip())
                            final_cat2 = new_name2.strip()
    
                        update_entry(eid, {
                            "date": edate.isoformat(),
                            "period": to_period(edate),
                            "type": etype2,
                            "category": final_cat2,
                            "source": esource.strip(),
                            "amount": float(amt2),
                            "status": estatus,
                            "notes": enotes.strip(),
                            "is_manual": int(row["is_manual"]),  # keep existing manual flag
                            "tags": etags.strip()
                        })
                        st.success("Updated.")
                        st.rerun()
    
                if colB.button("Delete", type="secondary", key=f"btn_delete_{eid}"):
                    delete_entry(eid)
                    st.warning("Deleted.")
                    st.rerun()
# Import
with tab["Import"]:
    st.subheader("Upload Excel/CSV")
    up = st.file_uploader("Upload file", type=["xlsx","xls","csv"])
    if up is not None:
        raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.write("Preview")
        st.dataframe(raw.head(20), use_container_width=True)
    
        cols = list(raw.columns)
    
        # Auto guesses
        g_date   = guess_column(cols, ["date", "transaction date", "posting date", "value date", "book date", "posted on"])
        g_month  = guess_column(cols, ["month", "period", "statement month"])
        g_type   = guess_column(cols, ["type", "transaction type", "txn type", "debit/credit", "dr/cr"])
        g_cat    = guess_column(cols, ["category", "cat", "category name"])
        g_source = guess_column(cols, ["source", "payee", "description", "merchant", "narration", "memo", "details", "name"])
        g_amount = guess_column(cols, ["amount", "transaction amount", "amt", "value"])
        g_status = guess_column(cols, ["status", "state"])
        g_desc   = guess_column(cols, ["description", "memo", "details", "narration", "note", "notes"])
    
        st.info("Map columns. Auto-guesses are preselected; adjust if needed.")
        opt = ["<None>"] + cols
        c1, c2, c3 = st.columns(3)
        col_date  = c1.selectbox("Date column (or None)", opt, index=select_index_with_guess(opt, g_date))
        col_month = c2.selectbox("Month name column (or None)", opt, index=select_index_with_guess(opt, g_month))
        col_type  = c3.selectbox("Type column (Income/Expense)", opt, index=select_index_with_guess(opt, g_type))
    
        c4, c5, c6 = st.columns(3)
        col_cat    = c4.selectbox("Category column", opt, index=select_index_with_guess(opt, g_cat))
        col_source = c5.selectbox("Source/Payee column", opt, index=select_index_with_guess(opt, g_source))
        col_amount = c6.selectbox("Amount column", opt, index=select_index_with_guess(opt, g_amount))
    
        c7, c8 = st.columns(2)
        col_status = c7.selectbox("Status column (optional)", opt, index=select_index_with_guess(opt, g_status))
        col_desc   = c8.selectbox("Description/Notes column (optional)", opt, index=select_index_with_guess(opt, g_desc))
    
        if col_month != "<None>":
            year_for_month = st.number_input("Year for the 'Month' column", min_value=2000, max_value=2100, value=date.today().year)
        else:
            year_for_month = date.today().year
    
        auto_mark_manual = st.checkbox("Mark imported as manual?", value=False)
        do_autocat = st.checkbox("Auto-categorize missing expense categories", value=True)
    
        st.markdown("---")
        st.subheader("Step 2: Preview, detect duplicates/conflicts, and choose actions")
    
        if st.button("Preview & detect matches"):
            # Normalize rows (but don't commit yet)
            rules = load_rules()
            processed = []
            for _, r in raw.iterrows():
                # Amount
                try:
                    amount_val = float(r[col_amount]) if col_amount != "<None>" else 0.0
                except Exception:
                    amount_val = 0.0
    
                # Type
                type_ = (str(r[col_type]).strip().title() if col_type != "<None>" and not pd.isna(r[col_type]) else "")
                if type_ not in ("Income","Expense"):
                    type_ = "Expense" if amount_val < 0 else "Income"
    
                # Status
                status = (str(r[col_status]).title() if (col_status != "<None>" and not pd.isna(r[col_status])) else "Paid")
    
                # Source / Category / Notes
                category = (str(r[col_cat]).strip() if col_cat != "<None>" and not pd.isna(r[col_cat]) else "")
                source   = (str(r[col_source]).strip() if col_source != "<None>" and not pd.isna(r[col_source]) else "")
                notes    = (str(r[col_desc]).strip() if col_desc != "<None>" and not pd.isna(r[col_desc]) else "")
    
                # Date / Period
                if col_date != "<None>" and not pd.isna(r[col_date]):
                    try:
                        dt = pd.to_datetime(r[col_date]).date()
                    except Exception:
                        dt = date(year_for_month, 1, 1)
                    period = to_period(dt)
                elif col_month != "<None>" and not pd.isna(r[col_month]):
                    mname = str(r[col_month])
                    m = month_name_to_num(mname)
                    m = m if m >= 1 else date.today().month
                    dt = date(year_for_month, m, 1)
                    period = f"{year_for_month:04d}-{m:02d}"
                else:
                    dt = date.today()
                    period = to_period(dt)
    
                # Normalize amount: store as positive number in DB, type tells direction
                amount_clean = abs(float(amount_val))
    
                row = {
                    "date": dt.isoformat(),
                    "period": period,
                    "type": type_,
                    "category": category,
                    "source": source,
                    "amount": amount_clean,
                    "status": status,
                    "notes": notes,
                    "is_manual": 1 if auto_mark_manual else 0,
                    "tags": ""
                }
    
                # Auto-categorize only if Expense with missing/empty category
                if do_autocat and type_ == "Expense" and (not category or category.lower() in ("", "nan", "none", "uncategorized")):
                    row["category"] = auto_categorize(row, rules)
    
                processed.append(row)
    
            if not processed:
                st.warning("No rows processed.")
            else:
                # Stage for review
                st.session_state["staged_import_rows"] = processed
                st.session_state["staged_import_filename"] = up.name
                st.success(f"Prepared {len(processed)} rows. Review below, then Apply import.")
                st.rerun()
    
        # If staged, show review grid
        if "staged_import_rows" in st.session_state:
            staged = pd.DataFrame(st.session_state["staged_import_rows"])
            existing = load_entries_df()  # across all months
    
            # Simple duplicate/conflict detection
            # Duplicate: same period, type, source (or category), and same rounded amount
            staged["amount_r"] = staged["amount"].round(2)
            existing["amount_r"] = existing["amount"].round(2)
            existing["source_l"] = existing["source"].fillna("").str.lower()
            existing["category_l"] = existing["category"].fillna("").str.lower()
    
            def find_match_info(srow):
                period, type_, amt = srow["period"], srow["type"], srow["amount_r"]
                src = str(srow.get("source","")).strip().lower()
                cat = str(srow.get("category","")).strip().lower()
    
                cand = existing[(existing["period"]==period) & (existing["type"]==type_)]
                # Duplicate if exact amount + (same source OR same category)
                dup = cand[((cand["amount_r"]==amt) & ((cand["source_l"]==src) | (cand["category_l"]==cat)))]
                if not dup.empty:
                    rid = int(dup.iloc[0]["id"])
                    return ("duplicate", rid, float(dup.iloc[0]["amount"]))
                # Conflict if same source OR same category but different amount
                conf = cand[((cand["source_l"]==src) | (cand["category_l"]==cat))]
                if not conf.empty:
                    rid = int(conf.iloc[0]["id"])
                    return ("conflict", rid, float(conf.iloc[0]["amount"]))
                return ("new", None, None)
    
            flags, match_ids, manual_amts = [], [], []
            for _, srow in staged.iterrows():
                rel, mid, mamt = find_match_info(srow)
                flags.append(rel)
                match_ids.append(mid)
                manual_amts.append(mamt)
    
            staged["relation"] = flags      # new | duplicate | conflict
            staged["match_id"] = match_ids  # id of matched existing row (if any)
            staged["existing_amount"] = manual_amts
    
            # Default actions
            def default_action(rel):
                if rel == "duplicate":
                    return "skip"
                if rel == "conflict":
                    return "update_manual"  # update existing manual row to imported amount
                return "import_new"
            staged["action"] = staged["relation"].map(default_action)
    
            st.markdown("#### Review staged rows")
            st.caption("Choose an action per row. 'update_manual' will overwrite the matched row’s amount; 'skip' ignores the staged row; 'import_new' adds it as a new entry.")
            action_opts = ["import_new", "skip", "update_manual"]
    
            review_cols = ["date","period","type","source","category","amount","existing_amount","relation","action"]
            review_cfg = {
                "relation": st.column_config.SelectboxColumn("Relation", options=["new","duplicate","conflict"], disabled=True),
                "action": st.column_config.SelectboxColumn("Action", options=action_opts),
                "existing_amount": st.column_config.NumberColumn("Existing amount", format="%.2f", disabled=True),
                "amount": st.column_config.NumberColumn("Import amount", format="%.2f", disabled=True),
            }
    
            staged_view = st.data_editor(
                staged[review_cols],
                hide_index=True,
                use_container_width=True,
                column_config=review_cfg,
                key="staged_import_editor"
            )
    
            left, right = st.columns([1,1])
            if left.button("Apply import"):
                # Apply actions
                batch_id = add_import_batch(st.session_state.get("staged_import_filename","upload"))
                applied, skipped, updated = 0, 0, 0
    
                # Use the edited staged_view to get chosen actions
                for i, r in staged_view.iterrows():
                    action = r["action"]
                    original = staged.iloc[i].to_dict()
                    if action == "skip":
                        skipped += 1
                        continue
                    elif action == "update_manual" and not pd.isna(staged.iloc[i]["match_id"]):
                        mid = int(staged.iloc[i]["match_id"])
                        # Load the existing row to preserve all fields except amount/status/notes if desired
                        df_exist = existing[existing["id"]==mid]
                        if not df_exist.empty:
                            base = df_exist.iloc[0].to_dict()
                            # Update amount; keep other fields from existing
                            update_entry(mid, {
                                "date": base["date"],
                                "period": base["period"],
                                "type": base["type"],
                                "category": base["category"],
                                "source": base["source"],
                                "amount": float(original["amount"]),
                                "status": base["status"],  # or set to "Paid" if you want
                                "notes": (base.get("notes") or "") + " | reconciled",
                                "is_manual": int(base.get("is_manual", 0)),
                                "tags": (base.get("tags") or "")
                            })
                            updated += 1
                        else:
                            # no match found after all; import as new
                            original["import_batch_id"] = batch_id
                            add_entry(original)
                            applied += 1
                    else:
                        # import_new
                        original["import_batch_id"] = batch_id
                        add_entry(original)
                        applied += 1
    
                # Clear staged
                st.session_state.pop("staged_import_rows", None)
                st.session_state.pop("staged_import_filename", None)
                st.success(f"Import complete. Added {applied}, updated {updated}, skipped {skipped}.")
                st.rerun()
    
            if right.button("Discard staged import"):
                st.session_state.pop("staged_import_rows", None)
                st.session_state.pop("staged_import_filename", None)
                st.warning("Staged import discarded.")
                st.rerun()

    
# Templates
with tab["Templates"]:
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

        # Show items
        st.markdown("#### Template items")
        items = get_template_items(tid)
        if items.empty:
            st.info("No items yet.")
        else:
            st.dataframe(
                items[["id", "type", "category", "source", "amount", "status", "notes"]],
                use_container_width=True,
            )

        # Add item form (type-aware category + free-typed amount)
        st.markdown("Add item")
        with st.form(f"add_template_item_{tid}"):
            c1, c2, c3, c4 = st.columns(4)
            ttype = c1.selectbox("Type", ["Expense", "Income"], key=f"tmpl_type_{tid}")

            # Type-aware category dropdown
            cat_options = get_categories(ttype)
            cat_plus = cat_options + ["+ Add new category..."]
            sel_cat = c2.selectbox("Category", cat_plus, key=f"tmpl_cat_{tid}")

            # Amount as text input (robust typing)
            amt_text = c3.text_input("Amount", placeholder="e.g. 120.00", key=f"tmpl_amt_text_{tid}")
            tstatus = c4.selectbox("Status", ["Planned", "Pending", "Paid"], key=f"tmpl_status_{tid}")

            c5, c6 = st.columns([2, 2])
            tsource = c5.text_input("Source/Payee (optional)", key=f"tmpl_source_{tid}")
            tnotes = c6.text_input("Notes (optional)", key=f"tmpl_notes_{tid}")

            # Inline new category name when chosen
            new_cat_name = None
            if sel_cat == "+ Add new category...":
                new_cat_name = st.text_input("New category name", key=f"tmpl_newcat_{tid}")

            addit = st.form_submit_button("Add item")
            if addit:
                # Resolve category
                final_cat = sel_cat
                if sel_cat == "+ Add new category...":
                    if not new_cat_name or not new_cat_name.strip():
                        st.error("Please enter a new category name.")
                        st.stop()
                    if add_category(ttype, new_cat_name.strip()):
                        final_cat = new_cat_name.strip()
                    else:
                        st.error("Category already exists or invalid. Please choose another name.")
                        st.stop()

                # Parse amount
                amt_val = parse_amount_text(amt_text)
                if amt_val is None or amt_val < 0:
                    st.error("Please enter a valid non-negative amount.")
                    st.stop()

                add_template_item(
                    tid,
                    {
                        "type": ttype,
                        "category": final_cat,
                        "source": tsource.strip(),
                        "amount": float(amt_val),
                        "status": tstatus,
                        "notes": tnotes.strip(),
                    },
                )
                st.success("Item added.")
                # Clear only the amount/source/notes inputs to keep Type/Category for faster entry
                for k in (f"tmpl_amt_text_{tid}", f"tmpl_source_{tid}", f"tmpl_notes_{tid}"):
                    st.session_state.pop(k, None)
                st.rerun()

        # Apply template to month (de-duplicated list)
        st.markdown("#### Apply template to month")
        periods = existing_periods()
        # Ensure current sidebar month is included, but avoid duplicates
        periods = list(dict.fromkeys(periods + [st.session_state["period"]]))
        # Default to current sidebar month if present
        default_idx = periods.index(st.session_state["period"]) if st.session_state["period"] in periods else 0

        ap_period = st.selectbox("Month to apply", periods, index=default_idx, key=f"tmpl_apply_period_{tid}")
        ap_status = st.selectbox("Set status", ["Pending", "Planned", "Paid"], key=f"tmpl_apply_status_{tid}")
        ap_manual = st.checkbox("Mark as manual?", value=True, key=f"tmpl_apply_manual_{tid}")

        if st.button("Apply template", key=f"btn_apply_template_{tid}"):
            count = apply_template_to_month(tid, ap_period, set_status=ap_status, mark_manual=ap_manual)
            st.success(f"Applied {count} items to {ap_period}.")
            st.rerun()

        st.markdown("---")
        if st.button("Delete template", type="secondary", key=f"btn_delete_template_{tid}"):
            delete_template(tid)
            st.warning("Template deleted.")
            st.rerun()

    # Save current month as template
    st.markdown("#### Save current month as template")
    new_tname = st.text_input("Template name", value=f"{st.session_state['period']} plan", key="save_month_as_template_name")
    if st.button("Save month entries as template", key="btn_save_month_as_template"):
        dfm = load_entries_df(st.session_state["period"])
        if dfm.empty:
            st.info("No entries to save.")
        else:
            new_tid = add_template(new_tname.strip() or f"Template {now_ts()}")
            for _, r in dfm.iterrows():
                add_template_item(
                    new_tid,
                    {
                        "type": r["type"],
                        "category": r["category"],
                        "source": r["source"],
                        "amount": r["amount"],
                        "status": r["status"],
                        "notes": r.get("notes") or "",
                    },
                )
            st.success(f"Saved {len(dfm)} items to template '{new_tname}'.")
            st.rerun()

# Rules
if show_rules_tab:
    with tab["Rules"]:
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
with tab["Budgets"]:
    st.subheader(f"Budgets for {st.session_state['period']}")

    # Category dropdown (Expense categories only)
    exp_cats = get_categories("Expense")
    cat_plus = exp_cats + ["+ Add new category..."]

    c1, c2 = st.columns([2, 1.5])
    sel_budget_cat = c1.selectbox("Category", cat_plus, key="budget_cat_select")

    # Inline add new category
    if sel_budget_cat == "+ Add new category...":
        new_budget_cat = st.text_input("New expense category name", key="budget_new_cat_name")
        if st.button("Add category", key="btn_add_budget_cat"):
            if new_budget_cat and new_budget_cat.strip():
                if add_category("Expense", new_budget_cat.strip()):
                    st.session_state["budget_cat_select"] = new_budget_cat.strip()
                    st.success(f"Added '{new_budget_cat.strip()}' to Expense categories.")
                    st.rerun()
                else:
                    st.warning("That category already exists or name is invalid.")
            else:
                st.error("Please enter a category name.")

    # Amount as free-typed text (robust parsing)
    budget_amt_text = c2.text_input("Budget amount", value="", placeholder="e.g. 500.00", key="budget_amount_text")

    if st.button("Set/Update budget", key="btn_set_budget"):
        # Resolve category (handles "+ Add new...")
        if st.session_state["budget_cat_select"] == "+ Add new category...":
            new_cat_name = st.session_state.get("budget_new_cat_name", "")
            if not new_cat_name or not new_cat_name.strip():
                st.error("Please add a new category name or choose an existing one.")
                st.stop()
            add_category("Expense", new_cat_name.strip())
            final_cat = new_cat_name.strip()
        else:
            final_cat = st.session_state["budget_cat_select"]

        # Parse amount
        amt_val = parse_amount_text(budget_amt_text)
        if amt_val is None or amt_val < 0:
            st.error("Please enter a valid non-negative amount.")
        else:
            set_budget(st.session_state["period"], final_cat, float(amt_val))
            st.success(f"Budget set for {final_cat}: {amt_val:,.2f}")

            # Mini-patch: safely clear the amount input and rerun
            st.session_state.pop("budget_amount_text", None)
            st.rerun()

    st.markdown("### Current budgets")
    budgets_df = get_budgets(st.session_state["period"])
    if budgets_df.empty:
        st.info("No budgets set for this month yet.")
    else:
        st.dataframe(budgets_df, use_container_width=True)

        # Optional: quick delete
        with st.expander("Delete a budget line"):
            if not budgets_df.empty:
                opts = budgets_df.apply(lambda r: f"{r['category']} — {r['amount']}", axis=1).tolist()
                id_by_label = {f"{r['category']} — {r['amount']}": int(r["id"]) for _, r in budgets_df.iterrows()}
                sel_del = st.selectbox("Select budget to delete", [""] + opts, key="budget_delete_sel")
                if sel_del and st.button("Delete selected budget", key="btn_delete_budget"):
                    bid = id_by_label[sel_del]
                    conn = get_conn()
                    conn.execute("DELETE FROM budgets WHERE id=?", (bid,))
                    conn.commit()
                    conn.close()
                    st.warning("Budget deleted.")
                    st.rerun()





















