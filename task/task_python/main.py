import pandas as pd

# read csv file
df_classification = pd.read_csv('./data/heart.csv')
df_regression = pd.read_csv('./data/faang_stock_prices.csv')
df_clustering = pd.read_csv('./data/Customer_Behaviour.csv')

# kumpulan dataset
df_cols = {
    "classification": df_classification,
    "regression": df_regression,
    "clustering": df_clustering
}

# loop tiap dataset
for name, df in df_cols.items():
    print(f"\n===== DATA {name.upper()} =====")

    print("--- HEAD ---")
    print(df.head())

    print("--- DESCRIBE ---")
    print(df.describe())

    print("--- INFO ---")
    df.info()
