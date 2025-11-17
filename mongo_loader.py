import pandas as pd
import pymongo

def load_to_mongo():
    df = pd.read_csv("retail_cleaned.csv")

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["retail"]
    collection = db["sales"]

    records = df.to_dict(orient="records")
    collection.insert_many(records)

    print("✔ Loaded", len(records), "documents into MongoDB collection 'sales'")

if __name__ == "__main__":
    load_to_mongo()
