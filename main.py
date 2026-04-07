# main.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------------
# Data Loading
# -------------------------------
def load_data(base_path="Dataset"):
    fact_trans = pd.read_csv(f"{base_path}/FactTransaction.csv")
    dim_acc = pd.read_csv(f"{base_path}/DimAccount.csv")
    dim_cus = pd.read_csv(f"{base_path}/DimCustomer.csv")
    return fact_trans, dim_acc, dim_cus


# -------------------------------
# Data Integration
# -------------------------------
def merge_data(fact_trans, dim_acc, dim_cus):
    df = fact_trans.merge(dim_acc, on="AccountID")
    dim_cus = dim_cus.rename(columns={"Status": "CustomerStatus"})
    df = df.merge(dim_cus, on="CustomerID")
    return df


# -------------------------------
# Feature Engineering
# -------------------------------
def create_customer_features(df):
    customer_df = df.groupby("CustomerID").agg({
        "TransactionID": "count",
        "TransactionAmount": ["sum", "mean", "std"],
        "Balance": "mean"
    })

    customer_df.columns = [
        "TransactionID_count",
        "TransactionAmount_sum",
        "TransactionAmount_mean",
        "TransactionAmount_std",
        "Balance_mean"
    ]

    customer_df = customer_df.reset_index()

    # Channel usage
    channel_counts = df.pivot_table(
        index="CustomerID",
        columns="TransactionChannel",
        values="TransactionID",
        aggfunc="count",
        fill_value=0
    ).reset_index()

    customer_df = customer_df.merge(channel_counts, on="CustomerID")

    return customer_df


# -------------------------------
# Clustering
# -------------------------------
def perform_clustering(customer_df):
    features = customer_df[[
        "TransactionID_count",
        "TransactionAmount_mean",
        "TransactionAmount_std",
        "Balance_mean"
    ]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_df["Cluster"] = kmeans.fit_predict(scaled)

    return customer_df


# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    print("🚀 Starting Financial Behavior Pipeline...")

    fact_trans, dim_acc, dim_cus = load_data()
    df = merge_data(fact_trans, dim_acc, dim_cus)

    customer_df = create_customer_features(df)
    customer_df = perform_clustering(customer_df)

    print("✅ Pipeline completed successfully")
    print("\n📊 Sample Output:")
    print(customer_df.head())
    print(f"\nTotal Customers Processed: {len(customer_df)}")


if __name__ == "__main__":
    main()