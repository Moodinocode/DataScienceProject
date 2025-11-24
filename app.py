from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

RAW_PATH = Path("Online Retail.xlsx")
CLEAN_XLSX = Path("Online Retail Cleaned.xlsx")
CLEANING_REPORT = Path("artifacts/cleaning_report.json")
SUMMARY_PATH = Path("outputs/analysis_summary.json")
OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = OUTPUTS_DIR / "figures"
SQL_TOP_PRODUCTS = OUTPUTS_DIR / "sql_top_products.csv"
SQL_COUNTRY_REVENUE = OUTPUTS_DIR / "sql_country_revenue.csv"


st.set_page_config(
    page_title="E-commerce Customer Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Project 2 · Business & Consumer Analytics")
st.caption("Online Retail dataset · Data cleaning → SQL → Visualization → Modeling")


@st.cache_data
def load_clean_sample(limit: int = 1000) -> pd.DataFrame | None:
    if CLEAN_XLSX.exists():
        return pd.read_excel(CLEAN_XLSX, nrows=limit)
    return None


@st.cache_data
def load_cleaning_report() -> Dict | None:
    if CLEANING_REPORT.exists():
        return json.loads(CLEANING_REPORT.read_text())
    return None


@st.cache_data
def load_summary() -> Dict | None:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return None


@st.cache_data
def load_sql_tables() -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    top_products = pd.read_csv(SQL_TOP_PRODUCTS) if SQL_TOP_PRODUCTS.exists() else None
    country_revenue = (
        pd.read_csv(SQL_COUNTRY_REVENUE) if SQL_COUNTRY_REVENUE.exists() else None
    )
    return top_products, country_revenue


def get_code_snippet(path: Path, anchor: str, lines_after: int = 40) -> str:
    if not path.exists():
        return "File not found."
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines):
        if anchor in line:
            start = idx
            end = min(len(lines), idx + lines_after)
            return "\n".join(lines[start:end])
    return "Snippet not found."


with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        """
1. **Read** the UCI Online Retail Excel workbook.  
2. **Clean** & enrich with `data_preparation.py`.  
3. **Analyze & SQL** using `project2_analysis.py`.  
4. **Visualize** everything in this app.
"""
    )


st.header("1. Data Overview")
sample_df = load_clean_sample()
if sample_df is not None:
    st.write("Preview of the cleaned transactional dataset:")
    st.dataframe(sample_df.head(25))
else:
    st.warning("Cleaned dataset not found. Run `python data_preparation.py` first.")


st.header("2. Cleaning Summary")
clean_stats = load_cleaning_report()
if clean_stats:
    cols = st.columns(4)
    cols[0].metric("Rows (raw)", f"{clean_stats['rows_raw']:,}")
    cols[1].metric("Rows (clean)", f"{clean_stats['rows_clean']:,}")
    cols[2].metric("Missing CustomerIDs removed", f"{clean_stats['missing_customer_id']:,}")
    cols[3].metric("Duplicates removed", f"{clean_stats['duplicate_rows_removed']:,}")

    st.markdown("**Outlier handling**")
    outlier_cols = st.columns(2)
    outlier_cols[0].metric(
        "Qty cap threshold",
        f"{clean_stats['quantity_cap_threshold']:.0f}",
        f"-{clean_stats['quantity_values_capped']:,} adjusted",
    )
    outlier_cols[1].metric(
        "Price cap threshold (£)",
        f"{clean_stats['unitprice_cap_threshold']:.2f}",
        f"-{clean_stats['unitprice_values_capped']:,} adjusted",
    )

    with st.expander("See cleaning logic (from data_preparation.py)", expanded=False):
        snippet = get_code_snippet(Path("data_preparation.py"), "def load_and_clean", 120)
        st.code(snippet, language="python")
else:
    st.warning("Cleaning report missing. Re-run the data preparation script.")


st.header("3. KPI Highlights")
summary = load_summary()
if summary:
    top_customers = pd.DataFrame(summary["top_customers"])
    st.subheader("Top 10 customers by revenue")
    st.dataframe(top_customers)

    st.subheader("Basket pairing opportunities")
    st.table(pd.DataFrame(summary["top_bundles"]))

    st.markdown(
        f"""
- **Peak hour:** {summary['peak_hour']}h  
- **Peak weekday:** {summary['peak_weekday']}  
- **Monthly revenue window:** {summary['monthly_revenue_start']} → {summary['monthly_revenue_end']}
"""
    )
else:
    st.warning("Analysis summary missing. Run `python project2_analysis.py`.")


st.header("4. Visual Insights")
if summary and "figures" in summary:
    figure_info = summary["figures"]
    figure_descriptions = {
        "monthly_revenue": "Monthly Revenue Trend: tracks growth and seasonality.",
        "hourly_revenue": "Revenue by Hour: lunch-time spike suggests marketing opportunities.",
        "weekday_revenue": "Revenue by Weekday: Thursdays and Fridays drive most sales.",
        "top_products": "Top 10 Products: informs merchandising and stock planning.",
        "cohort_heatmap": "Cohort Heatmap: retention drops after month 3—focus churn campaigns.",
    }
    for key, caption in figure_descriptions.items():
        st.subheader(caption.split(":")[0])
        path_str = figure_info.get(key)
        if path_str and Path(path_str).exists():
            st.image(str(path_str), caption=caption, use_container_width=True)
        else:
            st.info(f"{key} figure not found.")
        with st.expander("Show plotting code"):
            if key == "cohort_heatmap":
                snippet = get_code_snippet(
                    Path("project2_analysis.py"), "def create_visualizations", 200
                )
            else:
                snippet = get_code_snippet(
                    Path("project2_analysis.py"), "def create_visualizations", 200
                )
            st.code(snippet, language="python")
else:
    st.warning("Figures not generated yet.")


st.header("5. SQL & Statistical Insights")
tp_df, country_df = load_sql_tables()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top products by revenue (SQL)")
    if tp_df is not None:
        st.dataframe(tp_df.head(10))
    else:
        st.info("sql_top_products.csv not found.")

with col2:
    st.subheader("Top countries by revenue (SQL)")
    if country_df is not None:
        st.dataframe(country_df)
    else:
        st.info("sql_country_revenue.csv not found.")

if summary and "hypothesis_test" in summary:
    ht = summary["hypothesis_test"]
    st.markdown(
        f"""
**Welch t-test (Average order value)**  
- {ht['country_a']} mean: £{ht['mean_a']:.2f}  
- {ht['country_b']} mean: £{ht['mean_b']:.2f}  
- *t* = {ht['t_stat']:.2f}, p-value = {ht['p_value']:.6f}
"""
    )
    with st.expander("Show hypothesis test code"):
        snippet = get_code_snippet(Path("project2_analysis.py"), "def hypothesis_test", 60)
        st.code(snippet, language="python")


st.header("6. Reproduce Pipeline")
st.markdown(
    """
```bash
python data_preparation.py   # cleaning + exports
python project2_analysis.py  # KPIs + figures + modeling
streamlit run app.py         # launch this dashboard
```
"""
)

