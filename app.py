
import streamlit as st
import pandas as pd
import numpy as np
import json
from io import BytesIO

st.set_page_config(page_title="Automated Data Cleaning Engine", layout="wide")


# ============================================================
# FILE LOADER
# ============================================================

def load_file(file):
    if file.name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            try:
                return pd.read_csv(file, encoding="latin1")
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding="cp1252")

    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    raise ValueError("Unsupported file type")


# ============================================================
# PROFILER AGENT - FINDS PROBLEMS BEFORE CLEANING
# ============================================================

def profiler_agent(df):
    issues = []

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    fully_empty_rows = int(df.isna().all(axis=1).sum())

    fully_empty_columns = [
        str(col) for col in df.columns
        if df[col].isna().all()
    ]

    unnamed_columns = [
        str(col) for col in df.columns
        if str(col).strip() == "" or "unnamed" in str(col).lower()
    ]

    duplicate_columns = [
        str(col) for col in df.columns[df.columns.duplicated()]
    ]

    high_missing_columns = [
        str(col) for col in df.columns
        if df[col].isna().mean() > 0.50
    ]

    possible_numeric_columns = []
    possible_date_columns = []
    mixed_type_columns = []

    for col in df.columns:
        series = df[col]

        cleaned_numeric = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
        )

        numeric_ratio = pd.to_numeric(cleaned_numeric, errors="coerce").notna().mean()
        date_ratio = pd.to_datetime(series, errors="coerce").notna().mean()

        if numeric_ratio >= 0.85 and df[col].dtype == "object":
            possible_numeric_columns.append(str(col))

        if date_ratio >= 0.70 and df[col].dtype == "object":
            possible_date_columns.append(str(col))

        if 0.30 < numeric_ratio < 0.85 and df[col].dtype == "object":
            mixed_type_columns.append(str(col))

    if total_missing > 0:
        issues.append(f"{total_missing} missing values found.")

    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate rows found.")

    if fully_empty_rows > 0:
        issues.append(f"{fully_empty_rows} fully empty rows found.")

    if fully_empty_columns:
        issues.append(f"{len(fully_empty_columns)} fully empty columns found: {fully_empty_columns}")

    if unnamed_columns:
        issues.append(f"{len(unnamed_columns)} unnamed columns found: {unnamed_columns}")

    if duplicate_columns:
        issues.append(f"{len(duplicate_columns)} duplicate column names found: {duplicate_columns}")

    if high_missing_columns:
        issues.append(f"{len(high_missing_columns)} columns have more than 50% missing values: {high_missing_columns}")

    if possible_numeric_columns:
        issues.append(f"{len(possible_numeric_columns)} text columns look numeric: {possible_numeric_columns}")

    if possible_date_columns:
        issues.append(f"{len(possible_date_columns)} text columns look like dates: {possible_date_columns}")

    if mixed_type_columns:
        issues.append(f"{len(mixed_type_columns)} columns may have mixed numeric/text values: {mixed_type_columns}")

    column_report = pd.DataFrame({
        "column": [str(col) for col in df.columns],
        "dtype": [str(df[col].dtype) for col in df.columns],
        "missing_count": [int(df[col].isna().sum()) for col in df.columns],
        "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
        "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        "sample_values": [
            ", ".join(df[col].dropna().astype(str).head(3).tolist())
            for col in df.columns
        ]
    })

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "fully_empty_rows": fully_empty_rows,
        "fully_empty_columns": fully_empty_columns,
        "unnamed_columns": unnamed_columns,
        "duplicate_columns": duplicate_columns,
        "high_missing_columns": high_missing_columns,
        "possible_numeric_columns": possible_numeric_columns,
        "possible_date_columns": possible_date_columns,
        "mixed_type_columns": mixed_type_columns,
        "issues_found": issues
    }

    return profile, column_report


# ============================================================
# CLEANER AGENT - CLEANS SAFE ISSUES
# ============================================================

def clean_column_name(col, fallback_name):
    col = str(col).strip()

    if col == "" or "unnamed" in col.lower():
        col = fallback_name

    col = col.lower()
    col = col.replace(" ", "_").replace("-", "_")

    cleaned = "".join(ch for ch in col if ch.isalnum() or ch == "_")

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")

    if cleaned == "":
        cleaned = fallback_name

    return cleaned


def make_unique_columns(columns):
    seen = {}
    final_columns = []

    for col in columns:
        if col not in seen:
            seen[col] = 0
            final_columns.append(col)
        else:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")

    return final_columns


def cleaner_agent(df):
    df_clean = df.copy()
    cleaning_log = []

    original_rows = df_clean.shape[0]
    original_columns = df_clean.shape[1]

    # Remove fully empty rows
    empty_rows = int(df_clean.isna().all(axis=1).sum())
    if empty_rows > 0:
        df_clean = df_clean.dropna(how="all")
        cleaning_log.append({
            "action": "Removed fully empty rows",
            "count": empty_rows,
            "risk": "safe"
        })

    # Remove fully empty columns
    empty_columns = [
        col for col in df_clean.columns
        if df_clean[col].isna().all()
    ]

    if empty_columns:
        df_clean = df_clean.drop(columns=empty_columns)
        cleaning_log.append({
            "action": "Removed fully empty columns",
            "columns": [str(c) for c in empty_columns],
            "risk": "safe"
        })

    # Preserve meaningful unnamed columns
    renamed_columns = []
    unknown_counter = 1

    for col in df_clean.columns:
        col_text = str(col).strip()

        if col_text == "" or "unnamed" in col_text.lower():
            sample_values = df_clean[col].dropna().astype(str)

            if len(sample_values) > 0:
                new_name = f"unknown_column_{unknown_counter}"
                unknown_counter += 1

                renamed_columns.append(new_name)

                cleaning_log.append({
                    "action": "Preserved unnamed column with meaningful data",
                    "old_column": str(col),
                    "new_column": new_name,
                    "sample_values": sample_values.head(5).tolist(),
                    "risk": "review"
                })
            else:
                new_name = f"empty_unknown_column_{unknown_counter}"
                unknown_counter += 1
                renamed_columns.append(new_name)
        else:
            renamed_columns.append(col_text)

    df_clean.columns = renamed_columns

    # Standardize column names
    old_columns = list(df_clean.columns)

    standardized_columns = [
        clean_column_name(col, f"column_{i + 1}")
        for i, col in enumerate(df_clean.columns)
    ]

    standardized_columns = make_unique_columns(standardized_columns)
    df_clean.columns = standardized_columns

    if old_columns != standardized_columns:
        cleaning_log.append({
            "action": "Standardized column names",
            "old_columns": [str(c) for c in old_columns],
            "new_columns": [str(c) for c in standardized_columns],
            "risk": "safe"
        })

    # Trim text and standardize null values
    object_columns = df_clean.select_dtypes(include=["object"]).columns

    for col in object_columns:
        before_missing = int(df_clean[col].isna().sum())

        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(
            ["", "nan", "NaN", "None", "NULL", "null", "N/A", "n/a"],
            pd.NA
        )

        after_missing = int(df_clean[col].isna().sum())

        cleaning_log.append({
            "action": "Trimmed text and standardized blank/null values",
            "column": col,
            "missing_before": before_missing,
            "missing_after": after_missing,
            "risk": "safe"
        })

    # Remove exact duplicate rows
    duplicate_rows = int(df_clean.duplicated().sum())

    if duplicate_rows > 0:
        df_clean = df_clean.drop_duplicates()
        cleaning_log.append({
            "action": "Removed exact duplicate rows",
            "count": duplicate_rows,
            "risk": "medium",
            "note": "For transaction datasets, review whether duplicates may be real repeated transactions."
        })

    # Convert obvious numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            numeric_candidate = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
            )

            converted = pd.to_numeric(numeric_candidate, errors="coerce")
            confidence = converted.notna().mean()

            if confidence >= 0.85:
                df_clean[col] = converted
                cleaning_log.append({
                    "action": "Converted text column to numeric",
                    "column": col,
                    "confidence": round(float(confidence), 3),
                    "risk": "safe"
                })

    # Convert obvious date columns
    for col in df_clean.columns:
        if "date" in col.lower() or "time" in col.lower():
            converted_date = pd.to_datetime(df_clean[col], errors="coerce")
            confidence = converted_date.notna().mean()

            if confidence >= 0.70:
                df_clean[col] = converted_date
                cleaning_log.append({
                    "action": "Converted column to datetime",
                    "column": col,
                    "confidence": round(float(confidence), 3),
                    "risk": "safe"
                })

    cleaning_summary = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "final_rows": int(df_clean.shape[0]),
        "final_columns": int(df_clean.shape[1]),
        "rows_removed": int(original_rows - df_clean.shape[0]),
        "columns_removed": int(original_columns - df_clean.shape[1]),
        "total_fixes_applied": len(cleaning_log),
        "cleaning_log": cleaning_log
    }

    return df_clean, cleaning_summary


# ============================================================
# VALIDATOR AGENT - CHECKS AFTER CLEANING
# ============================================================

def validator_agent(raw_df, clean_df):
    issues = []
    warnings = []

    raw_rows, raw_cols = raw_df.shape
    clean_rows, clean_cols = clean_df.shape

    if clean_df.empty:
        issues.append("Dataset became empty after cleaning.")

    if clean_cols == 0:
        issues.append("All columns were removed.")

    row_loss_percent = 0
    column_loss_percent = 0

    if raw_rows > 0:
        row_loss_percent = ((raw_rows - clean_rows) / raw_rows) * 100

    if raw_cols > 0:
        column_loss_percent = ((raw_cols - clean_cols) / raw_cols) * 100

    if row_loss_percent > 30:
        issues.append(f"High row loss: {row_loss_percent:.2f}% rows removed.")
    elif row_loss_percent > 10:
        warnings.append(f"Moderate row loss: {row_loss_percent:.2f}% rows removed.")

    if column_loss_percent > 60:
        issues.append(f"High column loss: {column_loss_percent:.2f}% columns removed.")
    elif column_loss_percent > 30:
        warnings.append(f"Moderate column loss: {column_loss_percent:.2f}% columns removed.")

    remaining_duplicates = int(clean_df.duplicated().sum())
    if remaining_duplicates > 0:
        warnings.append(f"{remaining_duplicates} duplicate rows still remain.")

    high_missing_columns = [
        str(col) for col in clean_df.columns
        if clean_df[col].isna().mean() > 0.70
    ]

    if high_missing_columns:
        warnings.append(f"Columns still having more than 70% missing values: {high_missing_columns}")

    unknown_columns = [
        str(col) for col in clean_df.columns
        if "unknown_column" in str(col)
    ]

    if unknown_columns:
        warnings.append(
            f"Meaningful unnamed columns were preserved and may need business labeling: {unknown_columns}"
        )

    quality_score = 100
    quality_score -= min(row_loss_percent * 0.8, 25)
    quality_score -= min(column_loss_percent * 0.7, 25)
    quality_score -= len(issues) * 20
    quality_score -= len(warnings) * 5

    quality_score = round(max(0, min(100, quality_score)), 2)

    if issues:
        status = "Needs Manual Review"
    elif quality_score >= 90:
        status = "Excellent"
    elif quality_score >= 75:
        status = "Good"
    elif quality_score >= 60:
        status = "Usable With Review"
    else:
        status = "Needs Manual Review"

    return {
        "status": status,
        "quality_score": quality_score,
        "issues_after_cleaning": issues,
        "warnings_after_cleaning": warnings,
        "row_loss_percent": round(float(row_loss_percent), 2),
        "column_loss_percent": round(float(column_loss_percent), 2),
        "remaining_duplicate_rows": remaining_duplicates
    }


# ============================================================
# BUILDER AGENT - EXPORTS CLEAN DATA AND REPORTS
# ============================================================

def builder_agent(clean_df, before_profile, after_profile, cleaning_summary, validation_report):
    full_report = {
        "before_cleaning_profile": before_profile,
        "cleaning_summary": cleaning_summary,
        "after_cleaning_profile": after_profile,
        "validation_report": validation_report
    }

    csv_output = clean_df.to_csv(index=False)
    json_output = json.dumps(full_report, indent=4, default=str)

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        clean_df.to_excel(writer, sheet_name="cleaned_data", index=False)

        pd.DataFrame(cleaning_summary["cleaning_log"]).to_excel(
            writer,
            sheet_name="cleaning_log",
            index=False
        )

        pd.DataFrame([validation_report]).to_excel(
            writer,
            sheet_name="validation",
            index=False
        )

    excel_buffer.seek(0)

    return {
        "csv": csv_output,
        "json_report": json_output,
        "excel": excel_buffer
    }


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("Automated Data Cleaning Engine")

st.write(
    "Upload a messy CSV or Excel file. The app will identify faults first, "
    "show them clearly, clean safe issues, validate the cleaned data, and provide downloads."
)

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        raw_df = load_file(uploaded_file)
        raw_copy = raw_df.copy()

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        # 1. BEFORE CLEANING PROFILE
        before_profile, before_column_report = profiler_agent(raw_df)

        st.subheader("Step 1: Faults Found Before Cleaning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", before_profile["rows"])
        c2.metric("Columns", before_profile["columns"])
        c3.metric("Missing Values", before_profile["total_missing_values"])
        c4.metric("Duplicate Rows", before_profile["duplicate_rows"])

        if before_profile["issues_found"]:
            st.warning("Problems found:")
            for issue in before_profile["issues_found"]:
                st.write(f"- {issue}")
        else:
            st.success("No major problems found before cleaning.")

        with st.expander("Column-Level Fault Report Before Cleaning"):
            st.dataframe(before_column_report, use_container_width=True)

        with st.expander("Full Before-Cleaning Profile JSON"):
            st.json(before_profile)

        # 2. CLEANING
        cleaned_df, cleaning_summary = cleaner_agent(raw_df)

        st.subheader("Step 2: Cleaning Actions Applied")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Removed", cleaning_summary["rows_removed"])
        c2.metric("Columns Removed", cleaning_summary["columns_removed"])
        c3.metric("Fixes Applied", cleaning_summary["total_fixes_applied"])

        with st.expander("Full Cleaning Log"):
            st.json(cleaning_summary)

        # 3. AFTER CLEANING PROFILE
        after_profile, after_column_report = profiler_agent(cleaned_df)

        st.subheader("Step 3: Faults Remaining After Cleaning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Rows", after_profile["rows"])
        c2.metric("Final Columns", after_profile["columns"])
        c3.metric("Remaining Missing Values", after_profile["total_missing_values"])
        c4.metric("Remaining Duplicate Rows", after_profile["duplicate_rows"])

        if after_profile["issues_found"]:
            st.warning("Remaining problems:")
            for issue in after_profile["issues_found"]:
                st.write(f"- {issue}")
        else:
            st.success("No major problems remain after cleaning.")

        with st.expander("Column-Level Report After Cleaning"):
            st.dataframe(after_column_report, use_container_width=True)

        # 4. VALIDATION
        validation_report = validator_agent(raw_copy, cleaned_df)

        st.subheader("Step 4: Validation Result")

        c1, c2 = st.columns(2)
        c1.metric("Quality Score", validation_report["quality_score"])
        c2.metric("Status", validation_report["status"])

        if validation_report["status"] in ["Excellent", "Good"]:
            st.success("Cleaned dataset is ready to use.")
        elif validation_report["status"] == "Usable With Review":
            st.warning("Dataset is usable, but review the warnings.")
        else:
            st.error("Manual review is recommended before using this dataset.")

        with st.expander("Validation Details"):
            st.json(validation_report)

        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head(50), use_container_width=True)

        # 5. EXPORTS
        outputs = builder_agent(
            clean_df=cleaned_df,
            before_profile=before_profile,
            after_profile=after_profile,
            cleaning_summary=cleaning_summary,
            validation_report=validation_report
        )

        st.subheader("Download Clean Outputs")

        st.download_button(
            "Download Cleaned CSV",
            outputs["csv"],
            "cleaned_dataset.csv",
            "text/csv"
        )

        st.download_button(
            "Download Cleaning Report JSON",
            outputs["json_report"],
            "data_cleaning_report.json",
            "application/json"
        )

        st.download_button(
            "Download Cleaned Excel Workbook",
            outputs["excel"],
            "cleaned_dataset_with_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error while processing file: {e}")

else:
    st.info("Upload a CSV or Excel file to begin.")
