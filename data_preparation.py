from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("Online Retail.xlsx")
CLEAN_CSV = Path("retail_cleaned.csv")
CLEAN_XLSX = Path("Online Retail Cleaned.xlsx")
CLEANING_REPORT = Path("artifacts/cleaning_report.json")


def _ensure_output_dirs() -> None:
    CLEANING_REPORT.parent.mkdir(parents=True, exist_ok=True)


def load_raw_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the original Excel workbook."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path.resolve()}")
    return pd.read_excel(path)


def load_and_clean(
    raw_path: Path = RAW_PATH,
    csv_path: Path = CLEAN_CSV,
    xlsx_path: Path = CLEAN_XLSX,
    report_path: Path = CLEANING_REPORT,
) -> pd.DataFrame:
    """
    Load the Online Retail dataset, clean it, and persist both CSV/XLSX versions.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset with helper time attributes.
    """

    _ensure_output_dirs()
    df = load_raw_dataset(raw_path).copy()
    metrics: dict[str, float | int] = {}

    metrics["rows_raw"] = len(df)
    metrics["missing_customer_id"] = int(df["CustomerID"].isna().sum())
    df = df.dropna(subset=["CustomerID"])

    metrics["missing_invoice_date"] = int(df["InvoiceDate"].isna().sum())
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    metrics["negative_or_zero_quantity"] = int((df["Quantity"] <= 0).sum())
    df = df[df["Quantity"] > 0]

    metrics["negative_or_zero_price"] = int((df["UnitPrice"] <= 0).sum())
    df = df[df["UnitPrice"] > 0]

    duplicate_mask = df.duplicated(
        subset=[
            "InvoiceNo",
            "StockCode",
            "InvoiceDate",
            "CustomerID",
            "Quantity",
            "UnitPrice",
        ],
        keep="first",
    )
    metrics["duplicate_rows_removed"] = int(duplicate_mask.sum())
    df = df[~duplicate_mask]

    df["CustomerID"] = df["CustomerID"].astype(int)

    country_before = df["Country"].astype(str)
    df["Country"] = country_before.str.strip().str.title()
    metrics["country_standardized"] = int((country_before != df["Country"]).sum())

    qty_cap = df["Quantity"].quantile(0.99)
    price_cap = df["UnitPrice"].quantile(0.99)
    metrics["quantity_cap_threshold"] = float(qty_cap)
    metrics["unitprice_cap_threshold"] = float(price_cap)
    metrics["quantity_values_capped"] = int((df["Quantity"] > qty_cap).sum())
    metrics["unitprice_values_capped"] = int((df["UnitPrice"] > price_cap).sum())

    df.loc[df["Quantity"] > qty_cap, "Quantity"] = qty_cap
    df.loc[df["UnitPrice"] > price_cap, "UnitPrice"] = price_cap

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"] = df["InvoiceDate"].dt.day
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour
    df["InvoiceWeekday"] = df["InvoiceDate"].dt.day_name()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    metrics["rows_clean"] = len(df)

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Cleaning complete. Saved {csv_path} and {xlsx_path}")
    print(f"Cleaning stats logged to {report_path}")
    return df


if __name__ == "__main__":
    cleaned = load_and_clean()
    print(cleaned.head())
