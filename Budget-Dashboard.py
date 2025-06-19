# Streamlit Budget Tracker with Enhancements
import streamlit as st
import pandas as pd
import os
from openpyxl import load_workbook
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Budget Tracker", layout="wide")

# Constants
DATA_FILE = Path("data/budget_tracker.xlsx")
TEMPLATE_SHEET_PREFIX = "Template_"
DEFAULT_PLACEHOLDER_SHEETS = {"Sheet", "Sheet1", "Temp"}

# Predefined dropdown options
type_options = ["Income", "Expense"]
category_options = [
    "Salary", "Rent", "Groceries", "Utilities", "Fuel", "Phone",
    "Internet", "Insurance", "Gym", "Camp", "Other"
]

# Ensure Excel file exists
def init_excel():
    if not DATA_FILE.exists():
        with pd.ExcelWriter(DATA_FILE, engine='openpyxl') as writer:
            df = pd.DataFrame(columns=["Type", "Category", "Description", "Amount"])
            df.to_excel(writer, sheet_name="Sheet1", index=False)

# Load workbook
def load_excel():
    return pd.ExcelFile(DATA_FILE)

# Remove default/placeholder sheet if other real sheets exist
def cleanup_placeholder_sheet():
    wb = load_workbook(DATA_FILE)
    sheets = wb.sheetnames
    placeholders = [s for s in sheets if s in DEFAULT_PLACEHOLDER_SHEETS]
    if placeholders and len(sheets) > len(placeholders):
        for ph in placeholders:
            del wb[ph]
        wb.save(DATA_FILE)

# Read month or template data
def read_sheet(sheet_name):
    xl = load_excel()
    if sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        for col in ["Type", "Category", "Description", "Amount"]:
            if col not in df.columns:
                df[col] = ""
        df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce').fillna(0.0)
        return df[["Type", "Category", "Description", "Amount"]]
    return pd.DataFrame(columns=["Type", "Category", "Description", "Amount"])

# Save DataFrame to sheet
def save_to_excel(sheet_name, df):
    with pd.ExcelWriter(DATA_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# UI Sidebar
st.sidebar.title("📊 Budget Tracker")
existing_sheets = load_excel().sheet_names if DATA_FILE.exists() else []
months = [s for s in existing_sheets if not s.startswith(TEMPLATE_SHEET_PREFIX)]
templates = [s.replace(TEMPLATE_SHEET_PREFIX, "") for s in existing_sheets if s.startswith(TEMPLATE_SHEET_PREFIX)]

selected_month = st.sidebar.selectbox("Select Month", ["+ New Month"] + months)

# Month creation
if selected_month == "+ New Month":
    new_month_name = st.sidebar.text_input("Enter new month name (e.g., July - 2025)")
    use_template = st.sidebar.checkbox("Use a template?")
    selected_template = st.sidebar.selectbox("Select template", ["None"] + templates) if use_template else None
    if st.sidebar.button("➕ Create Month"):
        if new_month_name:
            if use_template and selected_template and selected_template != "None":
                df_month = read_sheet(TEMPLATE_SHEET_PREFIX + selected_template)
                df_month["Amount"] = 0.0  # Clear all amount fields
            else:
                df_month = pd.DataFrame([
                    {"Type": "Income", "Category": "", "Description": "", "Amount": 0.0},
                    {"Type": "Expense", "Category": "", "Description": "", "Amount": 0.0}
                ], columns=["Type", "Category", "Description", "Amount"])

            save_to_excel(new_month_name, df_month)
            cleanup_placeholder_sheet()
            st.session_state["selected_month"] = new_month_name
            st.rerun()
else:
    st.session_state["selected_month"] = selected_month

# Active month sheet
current_month = st.session_state.get("selected_month")
if current_month and current_month != "+ New Month":
    st.header(f"📅 Editing: {current_month}")
    df = read_sheet(current_month)

    # Ensure correct data types
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Type"] = df["Type"].astype(str)
    df["Category"] = df["Category"].astype(str)
    df["Description"] = df["Description"].astype(str)

    # Display editable table with dropdowns
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    from st_aggrid.shared import GridUpdateMode

    # Build grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("Type", editable=True, cellEditor="agSelectCellEditor", cellEditorParams={'values': type_options})
    gb.configure_column("Category", editable=True, cellEditor="agSelectCellEditor", cellEditorParams={'values': category_options})
    gb.configure_column("Amount", editable=True, type=["numericColumn"], precision=0)
    gb.configure_column("Description", editable=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    # Define row coloring based on Type
    row_style_jscode = JsCode("""
    function(params) {
      if (params.data.Type === 'Income') { return {'backgroundColor':'#d4f8d4'}; }
      if (params.data.Type === 'Expense') { return {'backgroundColor':'#f8d4d4'}; }
    };
    """)
    gb.configure_grid_options(getRowStyle=row_style_jscode)

    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=400,
    )
    edited_df = grid_response['data']


    # Save manually to Excel
    if st.button("💾 Save to Excel"):
        save_to_excel(current_month, edited_df)
        st.success(f"Saved to {current_month} ✅")

    # Save as template
    with st.expander("🗂 Save as Template"):
        template_name = st.text_input("Template name")
        if st.button("📁 Save Template"):
            if template_name:
                df_template = edited_df.copy()
                df_template["Amount"] = 0.0  # Clear values
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

    # Optional: Bar chart
    st.subheader("📊 Breakdown by Category")
    if not edited_df.empty:
        category_summary = edited_df[edited_df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values(ascending=False)
        st.bar_chart(category_summary)

else:
    st.info("👈 Please select or create a month from the sidebar.")
