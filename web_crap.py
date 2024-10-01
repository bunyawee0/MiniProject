import requests
from bs4 import BeautifulSoup
import csv

maps = ['Bind', 'Haven', 'Split', 'Ascent', 'Icebox', 'Breeze', 'Pearl', 'Fracture', 'Lotus', 'Sunset', 'Abyss']
ranks = ['Placements', 'Iron', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Ascendant', 'Immortal', 'Radiant']

def get_agent_data(ranks):
    # url = f"https://www.metasrc.com/valorant/stats/agents"
    for rank in ranks:
        url = f'https://www.metasrc.com/valorant/stats/agents?ranks={rank}'

        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the webpage content
            soup = BeautifulSoup(response.content, 'html.parser')

            table = soup.select_one('table tbody')

            # If the table exists, extract rows
            if table:
                rows = table.find_all('tr')
                
                # Open a file to write data into
                with open(f'allmap_{rank}.csv', 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write headers (if available, replace with appropriate headers for the data)
                    writer.writerow(['Agent', 'Role', 'Tier', 'Score', 'Trend', 'Win %', 'Pick %', 'Dmg/Round', 'KDA'])  # Adjust the number of columns
                    
                    # Loop through each row and extract cell data
                    for row in rows:
                        cells = row.find_all('td')
                        data = [cell.text.strip() for cell in cells]
                        writer.writerow(data)  # Write row data to CSV file
                print(f'allmap_{rank}.csv file has been saved.')

def get_agent_map(map_name):
    for map_name in maps:
        url = f'https://www.metasrc.com/valorant/stats/agents/{map_name}'

                # Send a request to fetch the webpage content
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the webpage content
            soup = BeautifulSoup(response.content, 'html.parser')

            table = soup.select_one('table tbody')

            # If the table exists, extract rows
            if table:
                rows = table.find_all('tr')
                
                # Open a file to write data into
                with open(f'{map_name}.csv', 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write headers (if available, replace with appropriate headers for the data)
                    writer.writerow(['Agent', 'Role', 'Tier', 'Score', 'Trend', 'Win %', 'Pick %', 'Dmg/Round', 'KDA'])  # Adjust the number of columns
                    
                    # Loop through each row and extract cell data
                    for row in rows:
                        cells = row.find_all('td')
                        data = [cell.text.strip() for cell in cells]
                        writer.writerow(data)  # Write row data to CSV file
                print(f'{map_name}.csv file has been saved.')
            else:
                print(f'Table not found for {map_name}')
        else:
            print(f'Failed to retrieve the webpage for {map_name}. Status code: {response.status_code}')

def get_agent_rank(maps,ranks):
    for map_name in maps:
        for rank in ranks:
            # URL of the webpage
            url = f'https://www.metasrc.com/valorant/stats/agents/{map_name}?ranks={rank}'

            # Send a request to fetch the webpage content
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the webpage content
                soup = BeautifulSoup(response.content, 'html.parser')

                table = soup.select_one('table tbody')

                # If the table exists, extract rows
                if table:
                    rows = table.find_all('tr')
                    
                    # Open a file to write data into
                    with open(f'{map_name}_{rank}.csv', 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        
                        # Write headers (if available, replace with appropriate headers for the data)
                        writer.writerow(['Agent', 'Role', 'Tier', 'Score', 'Trend', 'Win %', 'Pick %', 'Dmg/Round', 'KDA'])  # Adjust the number of columns
                        
                        # Loop through each row and extract cell data
                        for row in rows:
                            cells = row.find_all('td')
                            data = [cell.text.strip() for cell in cells]
                            writer.writerow(data)  # Write row data to CSV file
                    print(f'{map_name}_{rank}.csv file has been saved.')
                else:
                    print(f'Table not found for {map_name}')
            else:
                print(f'Failed to retrieve the webpage for {map_name}. Status code: {response.status_code}')

if __name__ == "__main__":
    get_agent_data(ranks)