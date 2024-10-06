import pandas as pd

# Load the data
data = pd.read_csv('data.csv')

# Display the original data for reference
print("Original Data:")
print(data.head())

# Clean the Agent names: Remove extra characters and ensure proper formatting
data['Agent'] = data['Agent'].str.replace(r'(\w+)([A-Z])', r'\1 \2', regex=True)  # Add space before capital letters
data['Agent'] = data['Agent'].str.title()  # Capitalize the first letter of each word

# Remove duplicates (e.g., 'Astra Astra' to 'Astra') and keep first occurrence of other columns
data['Agent'] = data['Agent'].str.split().str[0]  # Take the first word only

# Drop 'Rank', 'Tier', 'Rank_x', 'Rank_y' columns
data = data.drop(columns=['Rank', 'Tier', 'Rank_x', 'Rank_y'], errors='ignore')

# Replace NaN and None values with the mean of each column
# Assuming numeric columns that need to be filled with their mean
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Percentage columns that need cleaning
percentage_columns = ['Win %', 'Pick %', 'Play %', 'Attacker Win %', 'Attacker KDA', 
                     'Defender Win %', 'Defender KDA', 'A Pick %', 'A Defuse %', 
                     'B Pick %', 'B Defuse %', 'C Pick %', 'C Defuse %']

# Convert percentage columns to float
for col in percentage_columns:
    # Convert the column to string type and replace empty strings with NaN
    data[col] = data[col].astype(str).replace('', float('nan'))  # Convert all to string
    data[col] = data[col].str.replace('%', '', regex=False).astype(float) / 100  # Remove '%' and convert to float
    data[col] = data[col].fillna(0)  # Fill NaNs with 0 after conversion

# Display the DataFrame with cleaned agent names and their corresponding data
print("Cleaned Data with Updated Agent Names:")
print(data)

# Save the cleaned data to a new CSV file
data.to_csv('data_clean.csv', index=False)
