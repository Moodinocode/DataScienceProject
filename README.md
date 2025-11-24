# Project 2 · E-commerce Customer Analytics

Business & Consumer Analytics project that walks through the full Intro to Data Science lifecycle using the UCI _Online Retail_ dataset.

## What’s Inside

- `data_preparation.py` – loads the Excel workbook, cleans/enriches it, and writes fresh CSV/XLSX artifacts plus cleaning stats.
- `project2_analysis.py` – produces customer KPIs, SQL-style aggregates (DuckDB), statistical tests, a repeat-purchase model, and saves charts to `outputs/`.
- `app.py` – Streamlit dashboard that showcases the cleaned data, KPI tables, SQL insights, and generated visuals (with code snippets).
- `Online Retail.xlsx` – untouched raw dataset (all other artifacts are generated on demand during a live demo).

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell
pip install -r requirements.txt  # or install packages below manually
```

Core packages: `pandas`, `numpy`, `openpyxl`, `duckdb`, `seaborn`, `matplotlib`, `scikit-learn`, `scipy`, `streamlit`.

## Workflow

1. **Read & Clean**

   ```bash
   python data_preparation.py
   ```

   Generates `retail_cleaned.csv`, `Online Retail Cleaned.xlsx`, and `artifacts/cleaning_report.json`.

2. **Analyze & Visualize**

   ```bash
   python project2_analysis.py
   ```

   Creates KPI tables and figures under `outputs/` (customer KPIs, SQL exports, PNG charts, model metrics, summary JSON).

3. **Present**
   ```bash
   streamlit run app.py
   ```
   Interactive GUI with:
   - Cleaned data preview & cleaning statistics
   - KPI tables (top customers, basket bundles, peak times)
   - Visual gallery (monthly revenue, hourly/weekday sales, top products, cohort heatmap)
   - SQL outputs & Welch t-test summary
   - Reproduction commands and inline code snippets for each stage

## Repository Etiquette

The repo intentionally ships only source code + the raw Excel. Before submission/hand-off:

- Delete regenerated folders (`outputs/`, `artifacts/`) and files (`retail_cleaned.csv`, `Online Retail Cleaned.xlsx`) to keep it minimal.
- Re-run the scripts live during presentation to showcase the pipeline.

## Optional Extensions

- Hook the DuckDB queries up to a real Postgres/MySQL instance using `sql_loader.py`.
- Expand the Streamlit app with additional tabs (e.g., cohort drill-downs or model diagnostics).
- Swap the logistic regression for a calibrated classifier and add cross-validation.

## References

- Dataset: [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- Course framing: “Introduction to Data Science – Read → Clean → Visualize → SQL/NoSQL → Real-world usage”.
