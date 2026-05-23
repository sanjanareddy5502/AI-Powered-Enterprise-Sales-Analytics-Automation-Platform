import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

st.set_page_config(page_title="AI Sales Analytics Platform", layout="wide")

st.title("AI-Powered Enterprise Sales Analytics & Automation Platform")
st.write("Upload a CSV or Excel sales dataset. The platform automatically cleans, analyzes, detects anomalies, segments customers, and generates business dashboards.")

uploaded_file = st.file_uploader("Upload your sales CSV or Excel file", type=["csv", "xlsx"])

def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def standardize_columns(df):
    mapping = {
        "Order Date": "order_date",
        "Ship Date": "ship_date",
        "Customer ID": "customer_id",
        "Customer Name": "customer_name",
        "Segment": "segment",
        "Country": "country",
        "City": "city",
        "State": "state",
        "Region": "region",
        "Category": "category",
        "Sub-Category": "sub_category",
        "Product Name": "product_name",
        "Sales": "sales",
        "Order ID": "order_id",
        "Product ID": "product_id"
    }
    df = df.rename(columns=mapping)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def clean_data(df):
    df = standardize_columns(df)
    df = df.drop_duplicates()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce", dayfirst=True)

    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        df = df.dropna(subset=["sales"])

    if "order_date" in df.columns:
        df["year"] = df["order_date"].dt.year
        df["month"] = df["order_date"].dt.month
        df["month_name"] = df["order_date"].dt.month_name()
        df["quarter"] = df["order_date"].dt.quarter

    return df

if uploaded_file is not None:
    df = load_file(uploaded_file)
    df = clean_data(df)

    st.success("File uploaded, cleaned, and processed successfully.")
    st.subheader("AI Business Insights Summary")

    st.markdown(f"""
    - Total revenue: **${df['sales'].sum():,.0f}**
    - Total records after cleaning: **{df.shape[0]:,}**
    - Average transaction value: **${df['sales'].mean():,.2f}**
    - Highest transaction value: **${df['sales'].max():,.2f}**
    """)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    required = ["sales", "order_date", "customer_name", "region", "category"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    total_sales = df["sales"].sum()
    total_orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
    total_customers = df["customer_name"].nunique()
    total_products = df["product_name"].nunique() if "product_name" in df.columns else 0

    st.subheader("Executive KPI Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Total Customers", f"{total_customers:,}")
    c4.metric("Total Products", f"{total_products:,}")

    st.subheader("Business Performance Dashboards")

    col1, col2 = st.columns(2)

    with col1:
        regional_sales = df.groupby("region")["sales"].sum().reset_index()
        fig_region = px.bar(regional_sales, x="region", y="sales", title="Regional Sales Performance")
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        category_sales = df.groupby("category")["sales"].sum().reset_index()
        fig_category = px.pie(category_sales, names="category", values="sales", title="Sales by Category")
        st.plotly_chart(fig_category, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly_sales = df.groupby(pd.Grouper(key="order_date", freq="ME"))["sales"].sum().reset_index()
    fig_monthly = px.line(monthly_sales, x="order_date", y="sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.subheader("AI Sales Forecasting Engine")

    forecast_data = monthly_sales.rename(
        columns={"order_date": "ds", "sales": "y"}
    )

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
            title="AI-Powered 6-Month Sales Forecast"
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        st.dataframe(forecast_display.tail(6))

        st.download_button(
            "Download Forecast Results",
            forecast_display.to_csv(index=False),
            "forecast_results.csv",
            "text/csv"
        )
    else:
        st.warning("Not enough monthly data for forecasting. At least 12 months are recommended.")

    st.subheader("AI-Based Anomaly Detection")

    anomaly_model = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = anomaly_model.fit_predict(df[["sales"]])
    df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

    anomalies = df[df["anomaly_label"] == "Anomaly"]

    st.metric("Anomalies Detected", len(anomalies))

    fig_anomaly = px.scatter(
        df,
        x="order_date",
        y="sales",
        color="anomaly_label",
        hover_data=["customer_name", "product_name"] if "product_name" in df.columns else ["customer_name"],
        title="AI-Based Sales Anomaly Monitoring"
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

    st.subheader("AI Customer Segmentation")

    customer_sales = df.groupby("customer_name")["sales"].sum().reset_index()

    scaler = StandardScaler()
    scaled_sales = scaler.fit_transform(customer_sales[["sales"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_sales["cluster"] = kmeans.fit_predict(scaled_sales)

    cluster_order = customer_sales.groupby("cluster")["sales"].mean().sort_values()
    segment_mapping = {
        cluster_order.index[0]: "Low Value",
        cluster_order.index[1]: "Medium Value",
        cluster_order.index[2]: "High Value"
    }

    customer_sales["segment_label"] = customer_sales["cluster"].map(segment_mapping)

    fig_segment = px.pie(
        customer_sales,
        names="segment_label",
        values="sales",
        title="Customer Value Segmentation"
    )
    st.plotly_chart(fig_segment, use_container_width=True)

    st.dataframe(customer_sales.sort_values("sales", ascending=False).head(20))

    st.subheader("Download Processed Outputs")

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False),
        "cleaned_sales_data.csv",
        "text/csv"
    )

    st.download_button(
        "Download Anomaly Report",
        anomalies.to_csv(index=False),
        "anomaly_report.csv",
        "text/csv"
    )

    st.download_button(
        "Download Customer Segments",
        customer_sales.to_csv(index=False),
        "customer_segments.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV or Excel file to begin.")