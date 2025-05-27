
import pandas as pd

# Load the CSV file
df = pd.read_csv("dz_regression/assets/energy_usage.csv")

print("Data loaded:")
print(df.head())  # print first few rows

# Just show some basic info
print("\nColumns and data types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())
