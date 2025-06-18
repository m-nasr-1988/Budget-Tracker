import streamlit as st
import pandas as pd
import os
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from datetime import datetime
import plotly.express as px

# === Config ===
DATA_PATH = "data/budget_tracker.xlsx"
SUMMARY_SHEET = "Yearly Summary"

# === Init Directories ===
os.makedirs("data", exist_ok=True)

# === Session state init ===
if "entries" not in st.session_state:
    st.session_state.entries = []
if "autosave" not in st.session_state:
    st.session_state.autosave = False
if "limits" not in st.session_state:
    st.session_state.limits = {}

# === Helper functions ===
def load_data():
    if os.path.exists(DATA_PATH):
        try:
            return load_workbook(DATA_PATH)
        except InvalidFileException:
            return None
    return None

def create_workbook():
    wb = load_workbook(DATA_PATH) if os.path.exists(DATA_PATH) else None
    if wb is None or SUMMARY_SHEET not in wb.sheetnames:
        df = pd.DataFrame(columns=["Month", "In", "Out", "Remaining"])
        with pd.ExcelWriter(DATA_PATH, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)
    return load_workbook(DATA_PATH)

def create_month_sheet(wb, month_name, copy_from=None):
    month_name = month_name.replace("Month-", "").strip()
    sheet_name = f"Month-{month_name}"
    if sheet_name not in wb.sheetnames:
        if copy_from:
            from_sheet = wb[f"Month-{copy_from}"]
            df = pd.DataFrame(from_sheet.values)
            df.columns = df.iloc[0]
            df = df.drop(0)
        else:
            df = pd.DataFrame(columns=["Date", "Description", "Category", "Type", "Amount"])

        with pd.ExcelWriter(DATA_PATH, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        st.success(f"✅ Created new sheet: {sheet_name}")
        wb = load_workbook(DATA_PATH)
    return wb

def update_summary(wb):
    data = []
    for sheet in wb.sheetnames:
        if sheet.startswith("Month-"):
            df = pd.read_excel(DATA_PATH, sheet_name=sheet)
            month = sheet.replace("Month-", "")
            total_in = df[df.Type == "Income"]["Amount"].sum()
            total_out = df[df.Type == "Expense"]["Amount"].sum()
            remaining = total_in - total_out
            data.append([month, total_in, total_out, remaining])
    df_sum = pd.DataFrame(data, columns=["Month", "In", "Out", "Remaining"])
    with pd.ExcelWriter(DATA_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_sum.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)

def get_months(wb):
    return [s.replace("Month-", "") for s in wb.sheetnames if s.startswith("Month-")]

# === Sidebar UI ===
st.sidebar.title("📅 Monthly Budget Tracker")
new_month = st.sidebar.text_input("Create a new month (e.g. July-2025)")
copy_from = st.sidebar.selectbox("Copy structure from (optional)", ["None"] + get_months(load_data() or create_workbook()))

if st.sidebar.button("➕ Create Month"):
    if new_month:
        wb = create_month_sheet(create_workbook(), new_month, None if copy_from == "None" else copy_from)
        st.rerun()

month_list = get_months(load_data() or create_workbook())
selected_month = st.sidebar.selectbox("Select Month", month_list)
sheet_name = f"Month-{selected_month}"

# === Spending Limit Setup ===
st.sidebar.subheader("💰 Monthly Budget Limit")
if selected_month not in st.session_state.limits:
    st.session_state.limits[selected_month] = 5000  # Default
limit = st.sidebar.number_input(
    f"Set limit for {selected_month}",
    min_value=0.0,
    value=float(st.session_state.limits[selected_month]),
    step=100.0
)

st.session_state.limits[selected_month] = limit

# === Main UI ===
st.title(f"📊 Budget Entry for {selected_month}")

with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", datetime.today())
        category = st.selectbox("Category", ["Salary", "Rent", "Fuel", "Utilities", "Internet", "Phone", "Cash/Saving", "Gym", "Expenses", "Other"])
        typ = st.radio("Type", ["Income", "Expense"])
    with col2:
        description = st.text_input("Description")
        amount = st.number_input("Amount", min_value=0.0, step=0.01)

    submitted = st.form_submit_button("Add Entry")
    if submitted:
        st.session_state.entries.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Description": description,
            "Category": category,
            "Type": typ,
            "Amount": amount
        })
        st.success("✅ Entry added (not yet saved)")
        if st.session_state.autosave:
            df_new = pd.DataFrame(st.session_state.entries)
            df_existing = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            with pd.ExcelWriter(DATA_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
            st.session_state.entries = []
            update_summary(load_data())
            st.success("✅ Autosaved to Excel.")

# === Display unsaved entries ===
if st.session_state.entries:
    st.subheader("📝 Unsaved Entries")
    st.dataframe(pd.DataFrame(st.session_state.entries))

    if st.button("💾 Save All Entries to Excel"):
        df_new = pd.DataFrame(st.session_state.entries)
        df_existing = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        with pd.ExcelWriter(DATA_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        update_summary(load_data())
        st.session_state.entries = []
        st.success("✅ All entries saved to Excel.")

# === Autosave toggle ===
st.sidebar.checkbox("Enable Autosave", value=False, key="autosave")

# === Display summary ===
if os.path.exists(DATA_PATH):
    st.subheader("📈 Yearly Summary")
    try:
        df_summary = pd.read_excel(DATA_PATH, sheet_name=SUMMARY_SHEET)
        st.dataframe(df_summary)
        st.bar_chart(df_summary.set_index("Month")[["In", "Out"]])
    except Exception as e:
        st.warning("Summary not available yet.")

    # === Enhancements ===
    st.subheader("📌 Category Breakdown for Selected Month")
    try:
        df_month = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
        df_exp = df_month[df_month.Type == "Expense"]
        if not df_exp.empty:
            cat_summary = df_exp.groupby("Category")["Amount"].sum().reset_index()
            fig = px.pie(cat_summary, names="Category", values="Amount", title="Expense Distribution")
            st.plotly_chart(fig)

        # Dynamic spending limit from sidebar
        total_out = df_exp["Amount"].sum()
        st.markdown(f"**Monthly Spending Limit:** ${limit:,.2f}")
        st.markdown(f"**Current Total Out:** ${total_out:,.2f}")
        if total_out > limit:
            st.error("🚨 You exceeded your budget limit!")
        else:
            st.success("✅ You are within the budget.")
    except Exception as e:
        st.warning("No data yet for this month to analyze.")
