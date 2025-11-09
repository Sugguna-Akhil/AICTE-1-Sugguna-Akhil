# Week 2: Data Preparation
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'data/sample_ev_data.csv'
df = pd.read_csv(DATA_PATH)
print('Loaded', len(df), 'rows.')
print(df.head())

print(df.info())
print(df.isnull().sum())

# Fill missing numeric columns
numeric_cols = ['battery_kwh','power_kw','range_km','weight_kg','year']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].median(), inplace=True)

# Fill missing brands
if 'brand' in df.columns:
    df['brand'] = df['brand'].fillna('Unknown')

print(df.describe())
df.to_csv('data/sample_ev_data_cleaned.csv', index=False)
print('✅ Cleaned dataset saved at data/sample_ev_data_cleaned.csv')
