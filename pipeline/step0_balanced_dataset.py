# step0_balance_dataset.py
import pandas as pd

# Load the original dataset
df = pd.read_csv("smart_grid_dataset.csv")

# Create the 'Fault' column: 1 if either fault condition is 1
df['Fault'] = ((df['Overload Condition'] == 1) | (df['Transformer Fault'] == 1)).astype(int)

# Split into faulty and normal
faulty_df = df[df['Fault'] == 1]
normal_df = df[df['Fault'] == 0]

# Use all faulty samples, and sample remaining from normal
faulty_count = len(faulty_df)
normal_sampled_df = normal_df.sample(n=20000 - faulty_count, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([faulty_df, normal_sampled_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced_df.to_csv("balanced_smart_grid_dataset.csv", index=False)
print("Balanced dataset saved with shape:", balanced_df.shape)
