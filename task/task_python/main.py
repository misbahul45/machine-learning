import pandas as pd

df = pd.read_csv('./data/Boston-house-price-data.csv')

print("========= Sample data =========")
print(df.head())


print("\n========= Sample data =========")
print(df.describe())