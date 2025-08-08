import pandas as pd
import numpy as np
import os

# Correct raw file path (use raw string to avoid escape sequence issues)
raw_data_path = r"E:\exl training\day19\capstone1\data\raw\exl_credit_card_churn_data.csv"
cleaned_data_path = r"E:\exl training\day19\capstone1\data\processed\cleaned3_credit_card_churn_data.csv"

# Check if file exists
if not os.path.exists(raw_data_path):
    raise FileNotFoundError(f"❌ File not found: {raw_data_path}")

# Load data
df = pd.read_csv(raw_data_path)
print("Initial rows:", len(df))
print(df.isnull().any())

# Clean Gender
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace('', np.nan)
df['Gender'] = df['Gender'].fillna('Unknown')
print("After gender cleaning:", len(df))
# astype(str) → Converts all values in the Gender column to strings (just in case there are numbers or NaNs).
# str.strip() → Removes extra spaces from the beginning and end of each string.
# str.capitalize() → Capitalizes the first letter of each gender value (e.g., "male" → "Male", "FEMALE" → "Female").

# Fix HasCrCard
# Converting categorical values into numerical format
df['HasCrCard'] = df['HasCrCard'].replace({'Yes': 1, 'No': 0})
df['HasCrCard'] = pd.to_numeric(df['HasCrCard'], errors='coerce').fillna(0)

# Fix IsActiveMember
# Fix IsActiveMember: Convert 'Yes'/'No' to 1/0, then ensure binary format
df['IsActiveMember'] = df['IsActiveMember'].replace({'Yes': 1, 'No': 0})
df['IsActiveMember'] = pd.to_numeric(df['IsActiveMember'], errors='coerce')
df['IsActiveMember'] = df['IsActiveMember'].fillna(0).clip(0, 1)

print("After HasCrCard & IsActiveMember fix:", len(df))

# .fillna(0)

# Fills missing (NaN) values with 0 → meaning "not active".

# .clip(0, 1)

# Restricts the values to the range between 0 and 1.

# If any values were wrongly set to things like -1 or 2, they will be clipped:

# Values < 0 become 0

# Values > 1 become 1
# Remove invalid Age/Salary
df = df[(df['Age'] > 0) & (df['Age'] <= 100)]
df = df[df['EstimatedSalary'] >= 0]
print("After Age & Salary filter:", len(df))

# Handle missing Balance
df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
df['Balance'] = df['Balance'].fillna(df['Balance'].median())

# Clean and convert Churn
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
df = df[df['Churn'].isin([0.0, 1.0])]
df['Churn'] = df['Churn'].astype(int)
print("After filtering Churn:", len(df))

# Save cleaned version
df.reset_index(drop=True, inplace=True)
df.to_csv(cleaned_data_path, index=False)

print(f"🎉 Cleaned data saved to: {cleaned_data_path}")
print("🧾 Final cleaned data shape:", df.shape)
