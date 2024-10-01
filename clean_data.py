import pandas as pd
import glob

# List all CSV files in the current directory
csv_files = glob.glob("*.csv")

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file and append its DataFrame to the list
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    parts = csv_file.replace('.csv', '').rsplit('_', 1)
    map_name = parts[0]  # The part before the last underscore is the map name
    rank = parts[1] if len(parts) > 1 else 'Allrank'  # The part after the last underscore is the rank, defaulting to 'Allrank'

    # Add 'Map' and 'Rank' columns to store the map name and rank
    df['Map'] = map_name
    df['Rank'] = rank
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame into a new CSV file
combined_df.to_csv('all_maps_ranks_combined.csv', index=False)

print("All CSV files have been successfully merged and saved as 'all_maps_ranks_combined.csv'.")
