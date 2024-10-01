import requests
from bs4 import BeautifulSoup
import csv
import os

# List of ranks to scrape
ranks = ['Placements', 'Iron', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Ascendant', 'Immortal', 'Radiant']

def get_agent_data(ranks):
    # Create the directory if it doesn't exist
    os.makedirs('data_map', exist_ok=True)

    # Iterate over each rank
    # for rank in ranks:
        # url = f"https://www.metasrc.com/valorant/stats/maps?ranks={rank}"
    url = f"https://www.metasrc.com/valorant/stats/maps"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the webpage content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract Map Stats
        map_stats_table = soup.select_one('table tbody')
        if map_stats_table:
            map_rows = map_stats_table.find_all('tr')
            
            # Write Map Stats to CSV
            with open(f'data_map/map.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write headers for Map Stats
                writer.writerow(['Map', 'Play %', 'Attacker Win %', 'Attacker KDA', 'Defender Win %', 'Defender KDA'])
                
                # Loop through each row and extract cell data for Map Stats
                for row in map_rows:
                    cells = row.find_all('td')
                    data = [cell.text.strip() for cell in cells]
                    writer.writerow(data)  # Write row data to CSV file
            print(f'map.csv file has been saved.')

        # Extract Plant Site Stats
        plant_stats_table = soup.find_all('table')[1].select_one('tbody')  # Select the second table
        if plant_stats_table:
            plant_rows = plant_stats_table.find_all('tr')
            
            # Write Plant Site Stats to a separate CSV file
            with open(f'data_map/plant_stats.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write headers for Plant Site Stats
                writer.writerow(['Map', 'Best Site', 'A Pick %', 'A Defuse %', 'B Pick %', 'B Defuse %', 'C Pick %', 'C Defuse %'])
                
                # Loop through each row and extract cell data for Plant Site Stats
                for row in plant_rows:
                    cells = row.find_all('td')
                    data = [cell.text.strip() for cell in cells]
                    writer.writerow(data)  # Write row data to CSV file
            print(f'plant_stats.csv file has been saved.')

get_agent_data(ranks)
