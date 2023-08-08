import pandas as pd

# Step 1: Read the CSV files
csv2_path = r'fits\corr.fits.csv'
df2 = pd.read_csv(csv2_path)

# Step 2: Perform the subtraction
# Subtract RA from field_ra
df2['result'] = df2['index_ra'] - df2['field_ra']

# Step 3: Keep only specific columns and the subtraction results
result_df = df2[['index_ra', 'field_ra', 'result']]

# Step 4: Sort the DataFrame by the absolute values of the 'result' column
result_df_sorted = result_df.iloc[result_df['result'].abs().argsort()]

# Step 5: Create a new CSV file with the sorted result
output_csv_path = r'fits\output_csv_sorted_abs.csv'
result_df_sorted.to_csv(output_csv_path, index=False)

print("Subtraction completed and sorted result (by absolute values) saved to:", output_csv_path)
