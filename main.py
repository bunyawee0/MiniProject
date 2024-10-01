import pandas as pd
import joblib

# โหลดโมเดลที่ฝึกไว้
model = joblib.load('model.pkl')

# ฟังก์ชันสำหรับการคาดการณ์ทีมชนะ
def predict_team_winner_from_training_data(training_data, agent_a_list, agent_b_list, map_name):
    # สร้าง DataFrame สำหรับข้อมูล input
    data = []

    for agent_a, agent_b in zip(agent_a_list, agent_b_list):
        # ดึงข้อมูลจาก DataFrame ที่ใช้ในการฝึกโมเดล
        entry_a = training_data[training_data['Agent'] == agent_a]
        entry_b = training_data[training_data['Agent'] == agent_b]

        # คำนวณค่าที่ต้องใช้ในการคาดการณ์
        if not entry_a.empty and not entry_b.empty:
            row_a = entry_a.iloc[0]
            row_b = entry_b.iloc[0]

            data.append({
                'Score': (row_a['Score'] + row_b['Score']) / 2,  # คำนวณคะแนนเฉลี่ย
                'Trend': row_a['Trend'],  # ใช้ Trend ของ Agent A
                'Pick %': (float(row_a['Pick %'].replace('%', '')) + float(row_b['Pick %'].replace('%', ''))) / 2,  # คำนวณ Pick %
                'Dmg/Round': (row_a['Dmg/Round'] + row_b['Dmg/Round']) / 2,  # คำนวณ Damage ต่อรอบเฉลี่ย
                'KDA': (row_a['KDA'] + row_b['KDA']) / 2,  # คำนวณ KDA เฉลี่ย
                'Map': map_name,
                'Agent A': agent_a,
                'Agent B': agent_b
            })

    # สร้าง DataFrame จากข้อมูลทั้งหมด
    df_input = pd.DataFrame(data)

    # แปลง Trend และ Map เป็นตัวเลข (ทำให้ตรงกับที่โมเดลคาดหวัง)
    df_input = pd.get_dummies(df_input, columns=['Trend', 'Map'], drop_first=True)

    # ตรวจสอบว่าฟีเจอร์ใน df_input ตรงกับที่โมเดลใช้หรือไม่
    expected_features = model.feature_names_in_
    missing_features = [feature for feature in expected_features if feature not in df_input.columns]
    
    # เพิ่มฟีเจอร์ที่ขาดหายไปด้วยค่าเริ่มต้น (ถ้าต้องการ)
    for feature in missing_features:
        df_input[feature] = 0

    # จัดเรียงคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง
    df_input = df_input[expected_features]

    # ทำการคาดการณ์
    predicted_win_rate = model.predict(df_input)

    # สร้างผลลัพธ์ว่าทีม A หรือ B ชนะ
    team_wins = []
    if predicted_win_rate.mean() >= 0.5:
        team_wins.append("ทีม A ชนะ")
    else:
        team_wins.append("ทีม B ชนะ")

    return team_wins[0], predicted_win_rate  # คืนค่าผลลัพธ์เป็นข้อความเดียวและอัตราการชนะ

if __name__ == '__main__':
    training_data = pd.read_csv('data_clean.csv')

    agent_a_list = input("กรุณากรอกชื่อ Agent Team A (คั่นด้วยเครื่องหมายจุลภาค): ").split(',')
    agent_b_list = input("กรุณากรอกชื่อ Agent Team B (คั่นด้วยเครื่องหมายจุลภาค): ").split(',')
    map_name = input("กรุณากรอกชื่อแผนที่: ")

    predicted_team_winner, predicted_win_rate = predict_team_winner_from_training_data(training_data, agent_a_list, agent_b_list, map_name)

    # แสดงผลการคาดการณ์
    print(f'ผลการคาดการณ์ทีมที่ชนะจากข้อมูลการฝึก: {predicted_team_winner}')

