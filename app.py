import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

st.set_page_config(page_title="AI Business Intelligence Platform", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

SEMANTIC_SCHEMA = {
    "sales": [
        "sales", "revenue", "amount", "sales amount", "total sales",
        "net sales", "gross revenue", "transaction value", "price",
        "total amount", "invoice amount"
    ],
    "order_date": [
        "order date", "sale date", "transaction date", "purchase date",
        "invoice date", "date", "created date", "billing date"
    ],
    "customer_name": [
        "customer name", "customer", "client", "buyer", "account name",
        "consumer", "customer full name"
    ],
    "customer_id": [
        "customer id", "client id", "buyer id", "account id"
    ],
    "category": [
        "category", "product category", "product type", "department",
        "segment", "class"
    ],
    "region": [
        "region", "state", "city", "territory", "location",
        "country", "market", "area"
    ],
    "product_name": [
        "product name", "item name", "product", "sku name", "item"
    ],
    "order_id": [
        "order id", "invoice id", "transaction id", "receipt id", "sale id"
    ],
    "quantity": [
        "quantity", "qty", "units sold", "quantity sold", "volume"
    ],
    "unit_price": [
        "unit price", "price per unit", "selling price", "item price"
    ],
    "profit": [
        "profit", "margin", "net profit", "gross profit"
    ],
    # QuickBooks / accounting report fields
    "vendor_name": [
        "vendor", "vendor name", "payee", "name", "merchant",
        "supplier", "company name"
    ],
    "transaction_type": [
        "type", "transaction type", "txn type", "transaction category"
    ],
    "transaction_number": [
        "num", "number", "transaction number", "check number", "ref number"
    ],
    "memo": [
        "memo", "description", "note", "transaction memo"
    ],
    "account": [
        "account", "account name", "bank account", "ledger account"
    ],
    "cleared_status": [
        "clr", "cleared", "cleared status", "reconciled"
    ],
    "debit": [
        "debit", "withdrawal", "money out", "expense", "charge"
    ],
    "credit": [
        "credit", "deposit", "money in", "income", "payment"
    ],
    "split": [
        "split", "category split", "account split", "distribution"
    ]
}

def normalize_col(col):
    return str(col).strip().lower().replace("_", " ").replace("-", " ")

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
    return pd.read_excel(file)

def infer_column_type(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric_series.notna().mean()

    date_series = pd.to_datetime(series, errors="coerce", dayfirst=True)
    date_ratio = date_series.notna().mean()

    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if date_ratio > 0.60:
        return "date"
    elif numeric_ratio > 0.70:
        return "numeric"
    elif unique_ratio > 0.80:
        return "identifier_or_text"
    else:
        return "categorical"

def semantic_column_mapper(df):
    candidate_matches = []

    schema_embeddings = {
        target: model.encode(examples)
        for target, examples in SEMANTIC_SCHEMA.items()
    }

    for col in df.columns:
        clean_col = normalize_col(col)
        col_embedding = model.encode([clean_col])
        inferred_type = infer_column_type(df[col])

        best_target = None
        best_score = -1

        for target, embeddings in schema_embeddings.items():
            score = cosine_similarity(col_embedding, embeddings).max()

            if inferred_type == "numeric" and target in ["sales", "quantity", "unit_price", "profit"]:
                score += 0.12

            if inferred_type == "date" and target == "order_date":
                score += 0.30

            if inferred_type == "categorical" and target in ["region", "category"]:
                score += 0.08

            if inferred_type == "identifier_or_text" and target in ["customer_name", "product_name", "order_id", "customer_id"]:
                score += 0.06

            if score > best_score:
                best_score = score
                best_target = target

        candidate_matches.append({
            "original_column": col,
            "target_column": best_target,
            "score": float(best_score),
            "type": inferred_type
        })

    mapping = {}
    used_targets = set()

    sorted_matches = sorted(candidate_matches, key=lambda x: x["score"], reverse=True)

    for match in sorted_matches:
        original = match["original_column"]
        target = match["target_column"]
        score = match["score"]

        if score >= 0.58 and target not in used_targets:
            mapping[original] = target
            used_targets.add(target)
        else:
            mapping[original] = normalize_col(original).replace(" ", "_")

    return mapping, candidate_matches
def detect_dataset_type(df):
    cols = [normalize_col(c) for c in df.columns]
    all_cols = " ".join(cols)

    has_sales = any(word in all_cols for word in ["sales", "revenue", "amount", "price", "profit"])
    has_date = any(word in all_cols for word in ["date", "time", "timestamp"])
    has_customer = any(word in all_cols for word in ["customer", "client", "buyer"])
    has_interaction = any(word in all_cols for word in ["interaction", "event", "click", "view", "cart", "purchase"])
    has_user = any(word in all_cols for word in ["user", "visitor", "session"])
    has_inventory = any(word in all_cols for word in ["stock", "inventory", "warehouse", "quantity"])
    has_finance = any(word in all_cols for word in ["expense", "cost", "budget", "profit", "loss"])
    has_quickbooks = any(word in all_cols for word in ["debit", "credit", "clr", "split", "memo", "vendor"])

    if has_quickbooks:
        return "QuickBooks / Financial Transaction Dataset"

    if has_sales and has_date:
        return "Sales Dataset"

    if has_user and has_interaction:
        return "Ecommerce Interaction Dataset"

    if has_inventory:
        return "Inventory Dataset"

    if has_finance:
        return "Financial Dataset"

    return "General Business Dataset"
def universal_data_cleaning(df):

    cleaning_report = {}

    cleaning_report["Original Rows"] = df.shape[0]
    cleaning_report["Original Columns"] = df.shape[1]

    # Remove fully empty rows
    df = df.dropna(how="all")

    # Remove fully empty columns only
    df = df.dropna(axis=1, how="all")

    # Handle unnamed/blank columns intelligently
    new_columns = []

    unnamed_counter = 1

    for col in df.columns:

        col_str = str(col).strip()

        # If column name missing or unnamed
        if (
            col is None
            or col_str == ""
            or "unnamed" in col_str.lower()
        ):

            sample_values = df[col].dropna().astype(str)

            # If column has meaningful data
            if len(sample_values) > 0:

                inferred_name = f"unknown_column_{unnamed_counter}"

                unnamed_counter += 1

                new_columns.append(inferred_name)

            else:
                # Truly useless column
                new_columns.append(f"drop_column_{unnamed_counter}")
                unnamed_counter += 1

        else:
            new_columns.append(col_str)

    df.columns = new_columns

    # Remove truly useless unnamed columns
    drop_cols = [c for c in df.columns if "drop_column_" in c]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean text columns
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:

        df[col] = df[col].astype(str).str.strip()

        df[col] = df[col].replace(
            ["nan", "None", "NULL", ""],
            pd.NA
        )

    # Intelligent datetime conversion
    for col in df.columns:

        if "date" in col or "time" in col:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                dayfirst=True
            )

    # Intelligent numeric conversion
    for col in df.columns:

        if df[col].dtype == "object":

            cleaned_numeric = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
            )

            converted = pd.to_numeric(
                cleaned_numeric,
                errors="coerce"
            )

            if converted.notna().mean() > 0.70:
                df[col] = converted

    cleaning_report["Final Rows"] = df.shape[0]
    cleaning_report["Final Columns"] = df.shape[1]

    cleaning_report["Rows Removed"] = (
        cleaning_report["Original Rows"]
        - cleaning_report["Final Rows"]
    )

    cleaning_report["Columns Removed"] = (
        cleaning_report["Original Columns"]
        - cleaning_report["Final Columns"]
    )

    return df, cleaning_report


def apply_quickbooks_mapping(df, semantic_mapping):
    """
    QuickBooks reports often contain useful unnamed columns.
    This function prevents the AI mapper from mislabeling accounting fields.
    """
    quickbooks_manual_mapping = {
        "unknown_column_1": "vendor_name",
        "name": "vendor_name",
        "vendor": "vendor_name",
        "type": "transaction_type",
        "date": "order_date",
        "num": "transaction_number",
        "memo": "memo",
        "account": "account",
        "clr": "cleared_status",
        "split": "split",
        "debit": "debit",
        "credit": "credit"
    }

    fixed_mapping = dict(semantic_mapping)

    for original_col, target_col in quickbooks_manual_mapping.items():
        if original_col in df.columns:
            fixed_mapping[original_col] = target_col

    return fixed_mapping


def create_financial_value_column(df):
    """
    Creates one analysis value column from sales/debit/credit fields.
    This keeps the rest of the dashboard working for QuickBooks files.
    """
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
        return df

    if "debit" in df.columns:
        df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)

    if "credit" in df.columns:
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)

    if "debit" in df.columns and "credit" in df.columns:
        # Absolute transaction value for BI charts/anomaly detection.
        df["sales"] = df["debit"].abs() + df["credit"].abs()
    elif "debit" in df.columns:
        df["sales"] = df["debit"].abs()
    elif "credit" in df.columns:
        df["sales"] = df["credit"].abs()

    return df

st.title("AI-Powered Universal Business Intelligence Platform")

st.write(
    "Upload a CSV or Excel business dataset. The system automatically detects schema, "
    "cleans data, generates dashboards, forecasting, anomaly detection, customer intelligence, "
    "and downloadable reports."
)

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

def run_ecommerce_behavior_engine(df):

    st.subheader("Ecommerce Behavior Analytics Engine")

    possible_user_cols = [col for col in df.columns if "user" in col or "customer" in col]
    possible_product_cols = [col for col in df.columns if "product" in col or "item" in col]
    possible_interaction_cols = [col for col in df.columns if "interaction" in col or "event" in col or "type" in col]
    possible_date_cols = [col for col in df.columns if "date" in col or "time" in col or "timestamp" in col]

    user_col = possible_user_cols[0] if possible_user_cols else None
    product_col = possible_product_cols[0] if possible_product_cols else None
    interaction_col = possible_interaction_cols[0] if possible_interaction_cols else None
    date_col = possible_date_cols[0] if possible_date_cols else None

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Interactions", f"{len(df):,}")
    c2.metric("Unique Users", f"{df[user_col].nunique():,}" if user_col else "N/A")
    c3.metric("Unique Products", f"{df[product_col].nunique():,}" if product_col else "N/A")
    c4.metric("Interaction Types", f"{df[interaction_col].nunique():,}" if interaction_col else "N/A")

    if interaction_col:
        st.subheader("Interaction Type Distribution")
        interaction_counts = df[interaction_col].value_counts().reset_index()
        interaction_counts.columns = ["interaction_type", "count"]

        fig_interaction = px.bar(
            interaction_counts,
            x="interaction_type",
            y="count",
            title="Interaction Type Breakdown"
        )
        st.plotly_chart(fig_interaction, use_container_width=True)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        trend = df.dropna(subset=[date_col]).groupby(pd.Grouper(key=date_col, freq="D")).size().reset_index(name="interactions")

        st.subheader("Daily Interaction Trend")
        fig_trend = px.line(
            trend,
            x=date_col,
            y="interactions",
            title="Daily Ecommerce Interaction Trend",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    if product_col:
        st.subheader("Top Products by Engagement")
        top_products = df[product_col].value_counts().head(10).reset_index()
        top_products.columns = ["product", "interaction_count"]

        fig_products = px.bar(
            top_products,
            x="product",
            y="interaction_count",
            title="Top Products by User Engagement"
        )
        st.plotly_chart(fig_products, use_container_width=True)

    if user_col:
        st.subheader("Top Users by Activity")
        top_users = df[user_col].value_counts().head(10).reset_index()
        top_users.columns = ["user", "interaction_count"]

        st.dataframe(top_users)
if uploaded_file is not None:

    # Load file
    df = load_file(uploaded_file)

    # Detect dataset type
    dataset_type = detect_dataset_type(df)

    st.subheader("Detected Dataset Type")
    st.info(dataset_type)

    # Universal cleaning
    df, cleaning_report = universal_data_cleaning(df)

    st.subheader("Data Cleaning Summary")
    st.json(cleaning_report)

    # Semantic schema detection
    semantic_mapping, detection_report = semantic_column_mapper(df)

    # QuickBooks/accounting reports need manual correction because useful columns
    # can be unnamed and fields like Type, Split, Debit, Credit are accounting-specific.
    semantic_mapping = apply_quickbooks_mapping(df, semantic_mapping)

    st.subheader("Detected Semantic Schema")
    st.json(semantic_mapping)

    with st.expander("View Schema Detection Confidence"):
        st.dataframe(pd.DataFrame(detection_report))

    # Rename columns
    df = df.rename(columns=semantic_mapping)

    # Create one common value column so the existing dashboard works for
    # QuickBooks debit/credit files as well as sales files.
    df = create_financial_value_column(df)

    st.success("File uploaded, cleaned, and processed successfully.")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Ecommerce flow
    if dataset_type == "Ecommerce Interaction Dataset":

        run_ecommerce_behavior_engine(df)

        st.stop()

    # Validate value column
    if "sales" not in df.columns:

        st.error(
            "No sales/revenue/debit/credit amount column detected. "
            "Upload a business dataset with at least one usable numeric value column."
        )

        st.stop()

    # KPI calculations
    total_sales = df["sales"].sum()

    total_orders = (
        df["order_id"].nunique()
        if "order_id" in df.columns
        else len(df)
    )

    entity_col = "customer_name" if "customer_name" in df.columns else "vendor_name" if "vendor_name" in df.columns else None

    total_customers = (
        df[entity_col].nunique()
        if entity_col else 0
    )

    total_products = (
        df["product_name"].nunique()
        if "product_name" in df.columns
        else 0
    )

    # AI Summary
    st.subheader("AI Business Insights Summary")

    st.markdown(f"""
    - Total revenue/value detected: **${total_sales:,.0f}**
    - Total records after cleaning: **{df.shape[0]:,}**
    - Average transaction value: **${df['sales'].mean():,.2f}**
    - Highest transaction value: **${df['sales'].max():,.2f}**
    """)

    # KPI Dashboard
    st.subheader("Executive KPI Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales / Value", f"${total_sales:,.0f}")
    c2.metric("Total Orders / Records", f"{total_orders:,}")
    c3.metric("Total Customers / Vendors", f"{total_customers:,}")
    c4.metric("Total Products", f"{total_products:,}")

    # Business dashboards
    st.subheader("Business Performance Dashboards")

    col1, col2 = st.columns(2)

    with col1:

        if "region" in df.columns:

            regional_sales = (
                df.groupby("region")["sales"]
                .sum()
                .reset_index()
            )

            fig_region = px.bar(
                regional_sales,
                x="region",
                y="sales",
                title="Regional Performance"
            )

            st.plotly_chart(fig_region, use_container_width=True)

    with col2:

        if "category" in df.columns:

            category_sales = (
                df.groupby("category")["sales"]
                .sum()
                .reset_index()
            )

            fig_category = px.pie(
                category_sales,
                names="category",
                values="sales",
                title="Category Performance"
            )

            st.plotly_chart(fig_category, use_container_width=True)

    # Monthly trends
    if "order_date" in df.columns:

        st.subheader("Monthly Trend Analysis")

        monthly_sales = (
            df.groupby(
                pd.Grouper(
                    key="order_date",
                    freq="ME"
                )
            )["sales"]
            .sum()
            .reset_index()
        )

        fig_monthly = px.line(
            monthly_sales,
            x="order_date",
            y="sales",
            title="Monthly Sales / Value Trend",
            markers=True
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

        # Forecasting
        st.subheader("AI Forecasting Engine")

        forecast_data = monthly_sales.rename(
            columns={
                "order_date": "ds",
                "sales": "y"
            }
        )

        forecast_data = forecast_data.dropna()

        if forecast_data.shape[0] >= 12:

            forecast_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )

            forecast_model.fit(forecast_data)

            future_dates = forecast_model.make_future_dataframe(
                periods=6,
                freq="ME"
            )

            forecast_results = forecast_model.predict(future_dates)

            forecast_display = forecast_results[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ]

            fig_forecast = px.line(
                forecast_display,
                x="ds",
                y=["yhat", "yhat_lower", "yhat_upper"],
                title="AI-Powered 6-Month Forecast"
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )

            st.dataframe(forecast_display.tail(6))

            st.download_button(
                "Download Forecast Results",
                forecast_display.to_csv(index=False),
                "forecast_results.csv",
                "text/csv"
            )

        else:

            st.warning(
                "Not enough monthly data for forecasting. "
                "At least 12 months are recommended."
            )

    # Anomaly detection
    st.subheader("AI-Based Anomaly Detection")

    anomalies = pd.DataFrame()

    if df.shape[0] >= 10:

        anomaly_model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

        df["anomaly"] = anomaly_model.fit_predict(df[["sales"]])

        df["anomaly_label"] = df["anomaly"].map({
            1: "Normal",
            -1: "Anomaly"
        })

        anomalies = df[df["anomaly_label"] == "Anomaly"]

        st.metric("Anomalies Detected", len(anomalies))

        x_axis = (
            "order_date"
            if "order_date" in df.columns
            else df.index
        )

        hover_cols = [
            col for col in [
                "customer_name",
                "vendor_name",
                "product_name",
                "category",
                "region",
                "transaction_type",
                "memo"
            ]
            if col in df.columns
        ]

        fig_anomaly = px.scatter(
            df,
            x=x_axis,
            y="sales",
            color="anomaly_label",
            hover_data=hover_cols,
            title="AI-Based Anomaly Monitoring"
        )

        st.plotly_chart(
            fig_anomaly,
            use_container_width=True
        )

    else:

        st.warning(
            "Not enough records for anomaly detection."
        )

    # Customer/vendor segmentation
    if entity_col:

        st.subheader("AI Customer / Vendor Segmentation")

        customer_sales = (
            df.groupby(entity_col)["sales"]
            .sum()
            .reset_index()
        )

        if customer_sales.shape[0] >= 3:

            scaler = StandardScaler()

            scaled_sales = scaler.fit_transform(
                customer_sales[["sales"]]
            )

            kmeans = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )

            customer_sales["cluster"] = kmeans.fit_predict(
                scaled_sales
            )

            cluster_order = (
                customer_sales.groupby("cluster")["sales"]
                .mean()
                .sort_values()
            )

            segment_mapping = {
                cluster_order.index[0]: "Low Value",
                cluster_order.index[1]: "Medium Value",
                cluster_order.index[2]: "High Value"
            }

            customer_sales["segment_label"] = (
                customer_sales["cluster"]
                .map(segment_mapping)
            )

            fig_segment = px.pie(
                customer_sales,
                names="segment_label",
                values="sales",
                title="Customer Value Segmentation"
            )

            st.plotly_chart(
                fig_segment,
                use_container_width=True
            )

            st.dataframe(
                customer_sales
                .sort_values("sales", ascending=False)
                .head(20)
            )

            st.download_button(
                "Download Customer Segments",
                customer_sales.to_csv(index=False),
                "customer_segments.csv",
                "text/csv"
            )

        else:

            st.warning(
                "Not enough customers for segmentation."
            )

    # AI Assistant
    st.subheader("AI Conversational Analytics Assistant")

    def generate_ai_answer(df, user_question):

        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        data_summary = {
            "columns": list(df.columns),
            "rows": df.shape[0],
            "sample_rows": df.head(5).to_dict(orient="records"),
            "numeric_summary": df.describe(include="number").to_dict()
        }

        prompt = f"""
        You are an AI business analyst.
        Use the dataset summary below to answer the user's question.
        Be direct, practical, and business-focused.
        Do not invent values that are not supported by the data.

        Dataset Summary:
        {data_summary}

        User Question:
        {user_question}
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text

    user_question = st.text_input(
        "Ask a business question about this dataset"
    )

    if st.button("Ask AI Analyst"):

        if user_question.strip():

            with st.spinner("AI Analyst is thinking..."):

                answer = generate_ai_answer(
                    df,
                    user_question
                )

                st.write(answer)

        else:

            st.warning("Please type a question first.")

    # Downloads
    st.subheader("Download Processed Outputs")

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False),
        "cleaned_business_data.csv",
        "text/csv"
    )

    if not anomalies.empty:

        st.download_button(
            "Download Anomaly Report",
            anomalies.to_csv(index=False),
            "anomaly_report.csv",
            "text/csv"
        )

else:

    st.info("Upload a CSV or Excel file to begin.")
