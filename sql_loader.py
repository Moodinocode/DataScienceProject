import pandas as pd
from sqlalchemy import create_engine

def load_to_postgres():
    df = pd.read_csv("retail_cleaned.csv")

    # Example: adjust with your credentials
    engine = create_engine("postgresql://postgres:password@localhost:5432/retail_db")

    df.to_sql(
        "retail_cleaned",
        engine,
        if_exists="replace",
        index=False
    )

    print("✔ Data loaded into PostgreSQL table: retail_cleaned")


if __name__ == "__main__":
    load_to_postgres()
