from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preparation import CLEAN_XLSX, load_and_clean

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _prepare_clean_data() -> pd.DataFrame:
    if not CLEAN_XLSX.exists():
        load_and_clean()
    return pd.read_excel(CLEAN_XLSX, parse_dates=["InvoiceDate"])


def best_customers(df: pd.DataFrame) -> pd.DataFrame:
    customer = (
        df.groupby("CustomerID")
        .agg(
            revenue=("TotalPrice", "sum"),
            invoices=("InvoiceNo", "nunique"),
            items=("Quantity", "sum"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )
    customer["avg_order_value"] = customer["revenue"] / customer["invoices"]
    customer["days_between_first_last"] = (
        customer["last_purchase"] - customer["first_purchase"]
    ).dt.days
    customer["repeat_customer"] = customer["invoices"] > 1
    customer = customer.sort_values(["revenue", "invoices"], ascending=False)
    customer.to_csv(OUTPUT_DIR / "customer_kpis.csv", index=False)
    return customer


def sales_time_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sales_by_hour = (
        df.groupby("InvoiceHour")["TotalPrice"]
        .sum()
        .reset_index()
        .sort_values("TotalPrice", ascending=False)
    )
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    sales_by_day = (
        df.groupby("InvoiceWeekday")["TotalPrice"]
        .sum()
        .reindex(weekday_order)
        .reset_index()
        .rename(columns={"InvoiceWeekday": "Weekday"})
    )
    sales_by_day["TotalPrice"] = sales_by_day["TotalPrice"].fillna(0)
    sales_by_hour.to_csv(OUTPUT_DIR / "sales_by_hour.csv", index=False)
    sales_by_day.to_csv(OUTPUT_DIR / "sales_by_weekday.csv", index=False)
    return {"hour": sales_by_hour, "weekday": sales_by_day}


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.set_index("InvoiceDate")["TotalPrice"]
        .resample("ME")
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "Month", "TotalPrice": "Revenue"})
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_revenue.csv", index=False)
    return monthly


def _period_to_int(period: pd.Period) -> int:
    return period.year * 12 + period.month


def cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["OrderMonth"] = work["InvoiceDate"].dt.to_period("M")
    first_purchase = (
        work.groupby("CustomerID")["OrderMonth"].min().rename("CohortMonth")
    )
    work = work.join(first_purchase, on="CustomerID")
    work["CohortIndex"] = (
        work["OrderMonth"].apply(_period_to_int)
        - work["CohortMonth"].apply(_period_to_int)
        + 1
    )

    cohort_data = (
        work.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth", columns="CohortIndex", values="CustomerID"
    )
    cohort_pivot.to_csv(OUTPUT_DIR / "cohort_table.csv")
    return cohort_pivot


def product_bundles(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    transactions = (
        df.dropna(subset=["Description"])
        .groupby("InvoiceNo")["Description"]
        .apply(lambda x: sorted(set(x)))
    )
    pair_counts: Dict[tuple, int] = {}
    for items in transactions:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = (items[i], items[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    bundle_df = (
        pd.DataFrame(
            [(f"{a} + {b}", count) for (a, b), count in pair_counts.items()],
            columns=["ProductPair", "Count"],
        )
        .sort_values("Count", ascending=False)
        .head(top_n)
    )
    bundle_df.to_csv(OUTPUT_DIR / "product_bundles.csv", index=False)
    return bundle_df


def hypothesis_test(
    df: pd.DataFrame, country_a: str = "United Kingdom", country_b: str = "Germany"
) -> Dict[str, float]:
    order_totals = (
        df.groupby(["Country", "InvoiceNo"])["TotalPrice"].sum().reset_index()
    )
    sample_a = order_totals[order_totals["Country"] == country_a]["TotalPrice"]
    sample_b = order_totals[order_totals["Country"] == country_b]["TotalPrice"]
    test_result = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return {
        "country_a": country_a,
        "country_b": country_b,
        "mean_a": float(sample_a.mean()),
        "mean_b": float(sample_b.mean()),
        "t_stat": float(test_result.statistic),
        "p_value": float(test_result.pvalue),
    }


def repeat_purchase_model(customer_df: pd.DataFrame) -> Dict[str, object]:
    feature_cols = [
        "revenue",
        "invoices",
        "items",
        "avg_order_value",
        "days_between_first_last",
    ]
    model_df = customer_df.dropna(subset=feature_cols).copy()
    X = model_df[feature_cols].values
    y = model_df["repeat_customer"].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    (OUTPUT_DIR / "repeat_purchase_report.json").write_text(
        json.dumps({"classification_report": report, "roc_auc": roc_auc}, indent=2)
    )
    return {"roc_auc": roc_auc, "classification_report": report}


def run_sql_queries(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    con = duckdb.connect()
    con.register("retail", df)
    queries = {
        "monthly_revenue": """
            SELECT date_trunc('month', InvoiceDate) AS month_start,
                   SUM(TotalPrice) AS revenue
            FROM retail
            GROUP BY 1
            ORDER BY 1
        """,
        "top_products": """
            SELECT Description, SUM(TotalPrice) AS revenue
            FROM retail
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 15
        """,
        "country_revenue": """
            SELECT Country, SUM(TotalPrice) AS revenue, COUNT(DISTINCT CustomerID) AS customers
            FROM retail
            GROUP BY 1
            ORDER BY revenue DESC
            LIMIT 15
        """,
    }
    sql_outputs: Dict[str, pd.DataFrame] = {}
    for name, query in queries.items():
        result = con.execute(query).df()
        result.to_csv(OUTPUT_DIR / f"sql_{name}.csv", index=False)
        sql_outputs[name] = result
    con.close()
    return sql_outputs


def _save_current_figure(filename: str) -> str:
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def create_visualizations(
    monthly: pd.DataFrame,
    sales_patterns: Dict[str, pd.DataFrame],
    sql_results: Dict[str, pd.DataFrame],
    cohorts: pd.DataFrame,
) -> Dict[str, str]:
    figure_paths: Dict[str, str] = {}

    # Monthly revenue trend
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["Month"], monthly["Revenue"], marker="o")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    figure_paths["monthly_revenue"] = _save_current_figure("monthly_revenue.png")

    # Hour of day bar
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=sales_patterns["hour"].sort_values("InvoiceHour"),
        x="InvoiceHour",
        y="TotalPrice",
        ax=ax,
        color="#1f77b4",
    )
    ax.set_title("Revenue by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Revenue (£)")
    figure_paths["hourly_revenue"] = _save_current_figure("revenue_by_hour.png")

    # Weekday bar
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=sales_patterns["weekday"],
        x="Weekday",
        y="TotalPrice",
        ax=ax,
        color="#ff7f0e",
    )
    ax.set_title("Revenue by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Revenue (£)")
    ax.tick_params(axis="x", rotation=30)
    figure_paths["weekday_revenue"] = _save_current_figure("revenue_by_weekday.png")

    # Top products revenue
    top_products = sql_results["top_products"].head(10).sort_values("revenue", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_products["Description"], top_products["revenue"], color="#2ca02c")
    ax.set_title("Top Products by Revenue")
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("")
    figure_paths["top_products"] = _save_current_figure("top_products.png")

    # Cohort heatmap
    cohort_plot = cohorts.copy()
    cohort_plot.index = cohort_plot.index.astype(str)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        cohort_plot,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Active Customers"},
        linewidths=0.5,
    )
    ax.set_title("Monthly Cohort Retention")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    figure_paths["cohort_heatmap"] = _save_current_figure("cohort_heatmap.png")

    return figure_paths


def _format_date_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
    return formatted


def save_summary(summary: Dict[str, object]) -> None:
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    df = _prepare_clean_data()
    customer_kpis = best_customers(df)
    sales_patterns = sales_time_patterns(df)
    monthly = monthly_revenue(df)
    cohorts = cohort_table(df)
    bundles = product_bundles(df)
    hypo = hypothesis_test(df)
    sql_results = run_sql_queries(df)
    model_metrics = repeat_purchase_model(customer_kpis)
    figure_paths = create_visualizations(monthly, sales_patterns, sql_results, cohorts)

    top_customer_records = _format_date_columns(
        customer_kpis.head(10), ["first_purchase", "last_purchase"]
    ).to_dict("records")

    summary = {
        "top_customers": top_customer_records,
        "peak_hour": int(sales_patterns["hour"].iloc[0]["InvoiceHour"]),
        "peak_weekday": sales_patterns["weekday"].loc[
            sales_patterns["weekday"]["TotalPrice"].idxmax(), "Weekday"
        ],
        "monthly_revenue_start": monthly["Month"].min().strftime("%Y-%m"),
        "monthly_revenue_end": monthly["Month"].max().strftime("%Y-%m"),
        "top_bundles": bundles.head(5).to_dict("records"),
        "hypothesis_test": hypo,
        "model_roc_auc": model_metrics["roc_auc"],
        "sql_tables": {k: len(v) for k, v in sql_results.items()},
        "cohort_months": cohorts.shape[0],
        "figures": figure_paths,
    }
    save_summary(summary)
    print("Analysis pipeline complete. See outputs/ for artifacts.")


if __name__ == "__main__":
    main()
