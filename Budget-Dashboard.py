# Streamlit Budget Tracker with Fixes
import streamlit as st
import pandas as pd
import os
from openpyxl import load_workbook
from pathlib import Path

st.set_page_config(page_title="Budget Tracker", layout="wide")

# Constants
DATA_FILE = Path("data/budget_tracker.xlsx")
TEMPLATE_SHEET_PREFIX = "Template_"

# Predefined dropdown options
type_options = ["Income", "Expense"]
category_options = ["Salary", "Rent", "Groceries", "Utilities", "Fuel", "Phone", "Internet", "Insurance", "Gym", "Camp", "Other"]

# Ensure Excel file exists
def init_excel():
    if not DATA_FILE.exists():
        with pd.ExcelWriter(DATA_FILE, engine='openpyxl') as writer:
            df = pd.DataFrame(columns=["Type", "Category", "Description", "Amount"])
            df.to_excel(writer, sheet_name="Sheet1", index=False)

# Load workbook
def load_excel():
    return pd.ExcelFile(DATA_FILE)

# Read month or template data
def read_sheet(sheet_name):
    xl = load_excel()
    if sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        # Fill missing columns if any
        for col in ["Type", "Category", "Description", "Amount"]:
            if col not in df.columns:
                df[col] = ""
        # Fix column types for Streamlit compatibility
        df = df.fillna({"Type": "Expense", "Category": "", "Description": "", "Amount": 0})
        df["Type"] = df["Type"].astype(str)
        df["Category"] = df["Category"].astype(str)
        df["Description"] = df["Description"].astype(str)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        return df[["Type", "Category", "Description", "Amount"]]
    return pd.DataFrame(columns=["Type", "Category", "Description", "Amount"])

# Save DataFrame to sheet
def save_to_excel(sheet_name, df):
    with pd.ExcelWriter(DATA_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Init Excel if needed
init_excel()

# Sidebar UI
st.sidebar.title("📊 Budget Tracker")
existing_sheets = load_excel().sheet_names if DATA_FILE.exists() else []

# Clean up sheet list
months = [s for s in existing_sheets if not s.startswith(TEMPLATE_SHEET_PREFIX) and s.lower() != "sheet1"]
templates = [s.replace(TEMPLATE_SHEET_PREFIX, "") for s in existing_sheets if s.startswith(TEMPLATE_SHEET_PREFIX)]

selected_month = st.sidebar.selectbox("Select Month", ["+ New Month"] + months)

# Month creation logic
if selected_month == "+ New Month":
    new_month_name = st.sidebar.text_input("Enter new month name (e.g., July - 2025)")
    use_template = st.sidebar.checkbox("Use a template?")
    selected_template = st.sidebar.selectbox("Select template", ["None"] + templates) if use_template else None
    if st.sidebar.button("➕ Create Month"):
        if new_month_name:
            if use_template and selected_template and selected_template != "None":
                df_month = read_sheet(TEMPLATE_SHEET_PREFIX + selected_template)
                df_month["Amount"] = 0  # Clear amount fields
            else:
                df_month = pd.DataFrame([
                    {"Type": "Income", "Category": "Salary", "Description": "", "Amount": 0},
                    {"Type": "Expense", "Category": "", "Description": "", "Amount": 0}
                ])
            save_to_excel(new_month_name, df_month)
            st.session_state["selected_month"] = new_month_name
            st.rerun()
else:
    st.session_state["selected_month"] = selected_month

# Main dashboard logic
current_month = st.session_state.get("selected_month")
if current_month and current_month != "+ New Month":
    st.header(f"📅 Editing: {current_month}")
    df = read_sheet(current_month)

    # Data Editor
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=type_options),
            "Category": st.column_config.SelectboxColumn("Category", options=category_options),
            "Amount": st.column_config.NumberColumn("Amount", format=",.0f", step=1.0),
            "Description": st.column_config.TextColumn("Description")
        },
        use_container_width=True
    )

    # Save manually to Excel
    if st.button("💾 Save to Excel"):
        save_to_excel(current_month, edited_df)
        st.success(f"Saved to {current_month} ✅")

    # Save as Template
    with st.expander("🗂 Save as Template"):
        template_name = st.text_input("Template name")
        if st.button("📁 Save Template"):
            if template_name:
                df_template = edited_df.copy()
                df_template["Amount"] = 0  # Reset values
                save_to_excel(TEMPLATE_SHEET_PREFIX + template_name, df_template)
                st.success(f"Saved template: {template_name}")

    # Live Summary
    col1, col2 = st.columns(2)
    with col1:
        total_in = edited_df[edited_df["Type"] == "Income"]["Amount"].sum()
        total_out = edited_df[edited_df["Type"] == "Expense"]["Amount"].sum()
        st.metric("💰 Total In", f"{total_in:,.0f}")
        st.metric("💸 Total Out", f"{total_out:,.0f}")
    with col2:
        st.metric("📉 Remaining", f"{(total_in - total_out):,.0f}")

    # Bar Chart
    st.subheader("📊 Breakdown by Category")
    if not edited_df.empty:
        category_summary = edited_df[edited_df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values(ascending=False)
        st.bar_chart(category_summary)

else:
    st.info("👈 Please select or create a month from the sidebar.")
