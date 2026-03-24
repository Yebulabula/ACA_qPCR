import pandas as pd
import numpy as np

# Load your DataFrame (assuming it's stored as 'df.csv')
df = pd.read_csv("df_dPCR_GB_2025.csv")  # Change to your actual file

# Count the number of samples per Target
target_counts = df["Target"].value_counts()
max_samples = target_counts.max()

# Create a balanced dataset
balanced_df = pd.DataFrame()

for target, count in target_counts.items():
    df_target = df[df["Target"] == target]  # Subset for the current target

    # Randomly sample (with replacement) to match the max_samples
    df_upsampled = df_target.sample(n=max_samples, replace=True, random_state=42)

    # Append to the new balanced dataset
    balanced_df = pd.concat([balanced_df, df_upsampled])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
balanced_df.to_csv("balanced_dataframe.csv", index=False)

print("Dataset balanced. Saved as 'balanced_dataframe.csv'.")
