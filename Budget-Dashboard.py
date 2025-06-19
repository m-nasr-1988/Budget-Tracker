# streamlit_budget_app.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from openpyxl import load_workbook

# Constants
EXCEL_PATH = "data/budget_tracker.xlsx"
TEMPLATE_SHEET = "Templates"

# Utility Functions

def load_excel():
    if os.path.exists(EXCEL_PATH):
        return pd.ExcelFile(EXCEL_PATH)
    else:
        wb = load_workbook(filename=None)
        wb.save(EXCEL_PATH)
        return pd.ExcelFile(EXCEL_PATH)

def save_month_to_excel(month, df):
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=month, index=False)


def read_month_data(month):
    xl = load_excel()
    if month in xl.sheet_names:
        return xl.parse(month)
    return pd.DataFrame(columns=["Type", "Description", "Amount"])


def save_template(template_name, df):
    df_copy = df.copy()
    df_copy["Amount"] = ""
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df_copy.to_excel(writer, sheet_name=template_name, index=False)


def read_template(template_name):
    xl = load_excel()
    if template_name in xl.sheet_names:
        df = xl.parse(template_name)
        df["Amount"] = ""  # Clear values
        return df
    return pd.DataFrame(columns=["Type", "Description", "Amount"])

# Format helpers
def format_currency(value):
    try:
        return "{:,}".format(int(value))
    except:
        return value

# App Start
st.set_page_config(page_title="Monthly Budget Tracker", layout="wide")
st.title("📊 Monthly Budget Tracker")

# Sidebar
st.sidebar.header("🔧 Manage Budget")
months_existing = load_excel().sheet_names

selected_month = st.sidebar.selectbox("Select or create a month", months_existing + ["+ New Month"])

if selected_month == "+ New Month":
    new_month_name = st.sidebar.text_input("Enter new month name (e.g., July - 2025)")
    use_template = st.sidebar.checkbox("Load from template?")
    template_list = [s for s in months_existing if s != TEMPLATE_SHEET]
    selected_template = st.sidebar.selectbox("Template to load", template_list) if use_template else None
    if st.sidebar.button("➕ Create Month"):
        if new_month_name:
            if use_template and selected_template:
                df_month = read_template(selected_template)
            else:
                df_month = pd.DataFrame(columns=["Type", "Description", "Amount"])
            save_month_to_excel(new_month_name, df_month)
            st.experimental_rerun()
        else:
            st.sidebar.warning("Please enter a valid month name.")

elif selected_month:
    df = read_month_data(selected_month)
    st.subheader(f"🗓️ Editing: {selected_month}")

    st.markdown("### 🧾 Monthly Entries")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"Amount": st.column_config.NumberColumn(format=",", step=1)}
    )

    total_in = edited_df[edited_df["Type"] == "Income"]["Amount"].sum()
    total_out = edited_df[edited_df["Type"] == "Expense"]["Amount"].sum()
    remaining = total_in - total_out

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total In", format_currency(total_in))
    col2.metric("💸 Total Out", format_currency(total_out))
    col3.metric("💼 Remaining", format_currency(remaining))

    st.markdown("---")
    col_save, col_template = st.columns([1, 2])
    
    with col_save:
        if st.button("💾 Save to Excel"):
            save_month_to_excel(selected_month, edited_df)
            st.success("Data saved successfully!")

    with col_template:
        with st.expander("💠 Save this as Template"):
            template_name = st.text_input("Enter template name")
            if st.button("Save Template"):
                if template_name:
                    save_template(template_name, edited_df)
                    st.success(f"Template '{template_name}' saved!")
                else:
                    st.warning("Please enter a name for the template")
