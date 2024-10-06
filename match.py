import pandas as pd

# อ่านข้อมูลจาก CSV
data = pd.read_csv('valorant_match_data7.csv')

# สร้างฟังก์ชันเพื่อเปลี่ยนชื่อทีมในแต่ละแมตช์
def rename_teams(df):
    unique_teams = df['team'].unique()
    if len(unique_teams) >= 2:
        df['team'] = df['team'].replace({unique_teams[0]: 'Team A', unique_teams[1]: 'Team B'})
        df['winner'] = df['winner'].replace({unique_teams[0]: 'Team A', unique_teams[1]: 'Team B'})
    return df

# นำฟังก์ชันมาใช้กับแต่ละแมตช์
data = data.groupby('match_id').apply(rename_teams)
data.to_csv('updated_match_data.csv', index=False)
# แสดงผลลัพธ์
print(data)
