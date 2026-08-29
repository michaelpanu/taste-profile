import pandas as pd

INPUT_CSV = "data/foodfacts.csv"
OUTPUT_CSV = "data/foodsfactsfinal.csv"

ID_COLUMN_NAME = "food_id"

df = pd.read_csv(INPUT_CSV)

# If the ID column already exists, remove it so we can recreate it cleanly
if ID_COLUMN_NAME in df.columns:
    df = df.drop(columns=[ID_COLUMN_NAME])

# Create IDs from 1 to number of rows
food_ids = range(1, len(df) + 1)

# Insert as the first column
df.insert(0, ID_COLUMN_NAME, food_ids)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved new CSV to: {OUTPUT_CSV}")
print(f"Created {len(df)} IDs.")