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

# Group by Agent and get the first occurrence of each unique agent
unique_agents_data = data.groupby('Agent').first().reset_index()

# Display the DataFrame with unique agents and their corresponding data
print("Unique Agents with Other Columns:")
print(unique_agents_data)

unique_agents_data.to_csv('data_clean.csv', index=False)