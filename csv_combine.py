import pandas as pd

# Read the CSV files
df1 = pd.read_csv("evaluation_results.csv")
df2 = pd.read_csv("GPT_Answers.csv")

# Merge on the common field "id"
merged_df = pd.merge(df1, df2, on="question", how="inner")

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("combined.csv", index=False)
