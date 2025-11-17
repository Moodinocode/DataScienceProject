import pandas as pd

def load_and_clean():
    # Load Excel file
    df = pd.read_excel("Online Retail.xlsx")

    # ---- CLEANING ----
    # Remove rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove negative quantities (refunds)
    df = df[df["Quantity"] > 0]

    # Remove zero/negative prices
    df = df[df["UnitPrice"] > 0]

    # Fix types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Create new columns
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Save cleaned version for SQL/Mongo
    df.to_csv("retail_cleaned.csv", index=False)

    print("✔ Cleaning complete. Saved retail_cleaned.csv")
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())
