import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Budget Tracker", layout="wide")

DATA_FILE = Path("data/budget_tracker.xlsx")
COLUMNS = ["Type", "Category", "Description", "Amount"]
TYPE_OPTIONS = ["Income", "Expense"]
CATEGORY_OPTIONS = ["Salary", "Rent", "Groceries", "Utilities", "Fuel", "Phone", "Internet", "Insurance", "Gym", "Camp", "Other"]

def init_excel():
    if not DATA_FILE.exists():
        with pd.ExcelWriter(DATA_FILE, engine="openpyxl") as writer:
            pd.DataFrame(columns=COLUMNS).to_excel(writer, sheet_name="Sheet1", index=False)

def read_sheet(sheet_name):
    xl = pd.ExcelFile(DATA_FILE)
    if sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df = df.reindex(columns=COLUMNS).fillna("")
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        return df
    return pd.DataFrame(columns=COLUMNS)

def save_to_excel(sheet_name, df):
    with pd.ExcelWriter(DATA_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Initialize file and load data
init_excel()
current_sheet = "Sheet1"
if "df" not in st.session_state:
    st.session_state.df = read_sheet(current_sheet)

st.title("📝 Budget Tracker")

# Add new row
if st.button("➕ Add Row"):
    new_row = {"Type": "Income", "Category": "", "Description": "", "Amount": 0.0}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)

# Show table
df = st.session_state.df.copy()
df.insert(0, "Row", range(1, len(df) + 1))  # Auto row numbers

# Configure ag-grid
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(editable=True)
gb.configure_column("Type", editable=True, cellEditor="agSelectCellEditor", cellEditorParams={"values": TYPE_OPTIONS})
gb.configure_column("Category", editable=True, cellEditor="agSelectCellEditor", cellEditorParams={"values": CATEGORY_OPTIONS})
gb.configure_column("Amount", type=["numericColumn", "numberColumnFilter"], precision=2)
grid_options = gb.build()

grid_return = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.VALUE_CHANGED,
    theme="material",
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True
)

# Save edits from table
if grid_return["data"] is not None:
    edited_df = grid_return["data"].drop(columns=["Row"])
    edited_df["Amount"] = pd.to_numeric(edited_df["Amount"], errors="coerce").fillna(0.0)
    st.session_state.df = edited_df

# Save to file
if st.button("💾 Save"):
    save_to_excel(current_sheet, st.session_state.df)
    st.success("Saved ✅")
