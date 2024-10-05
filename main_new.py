import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# โหลดโมเดล พร้อม custom objects
model = load_model('cnn_model.h5', custom_objects={'mse': MeanSquaredError()})
# โหลดโมเดลที่ฝึกไว้
# model = joblib.load('model.pkl')

# ฟังก์ชันสำหรับการคาดการณ์ทีมชนะ
def predict_team_winner(training_data, agent_a_list, agent_b_list, map_name):
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
                'Score': (row_a['Score'] + row_b['Score']) / 2,
                'Trend': row_a['Trend'],
                'Pick %': (float(row_a['Pick %'].replace('%', '')) + float(row_b['Pick %'].replace('%', ''))) / 2,
                'Dmg/Round': (row_a['Dmg/Round'] + row_b['Dmg/Round']) / 2,
                'KDA': (row_a['KDA'] + row_b['KDA']) / 2,
                'Map': map_name,
                'Agent A': agent_a,
                'Agent B': agent_b
            })

    # สร้าง DataFrame จากข้อมูลทั้งหมด
    df_input = pd.DataFrame(data)

    # แปลง Trend และ Map เป็นตัวเลข (ทำให้ตรงกับที่โมเดลคาดหวัง)
    df_input = pd.get_dummies(df_input, columns=['Trend'], drop_first=True)

    # ตรวจสอบฟีเจอร์ทั้งหมดที่โมเดลคาดหวัง (ใช้ฟีเจอร์ที่ถูกต้องจากโมเดล)
    expected_features = ['Score', 'Pick %', 'Dmg/Round', 'KDA'] 

    # ตรวจสอบและจัดเรียงคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง
    df_input = df_input[expected_features]

    df_input = np.zeros((len(df_input), 28, 1))  

    # ทำการคาดการณ์
    predicted_win_rate = model.predict(df_input)

    # สร้างผลลัพธ์ว่าทีม A หรือ B ชนะ
    team_wins = []
    if predicted_win_rate[0][0] >= 0.5:
        team_wins = "Team A Win"
    else:
        team_wins = "Team B Win"

    return team_wins, predicted_win_rate.mean(), df_input

if __name__ == '__main__':
    training_data = pd.read_csv('data_clean.csv')
    agent_a_list = ['Killjoy','Brimstone','Raze','Sova','Breach']
    agent_b_list = ['Breach','Killjoy','Raze','Brimstone','Viper']
    map_name =' Fracture'
    # agent_a_list = input("กรุณากรอกชื่อ Agent Team A (คั่นด้วยเครื่องหมายจุลภาค): ").split(',')
    # agent_b_list = input("กรุณากรอกชื่อ Agent Team B (คั่นด้วยเครื่องหมายจุลภาค): ").split(',')
    # map_name = input("กรุณากรอกชื่อแผนที่: ")

    predicted_team_winner, predicted_win_rate , dfinput= predict_team_winner(training_data, agent_a_list, agent_b_list, map_name)
    # แสดงผลการคาดการณ์
    # print(predicted_win_rate)/

    print(f'ผลการคาดการณ์ทีมที่ชนะ: {predicted_team_winner}')
    predicted_win_rate = model.predict(dfinput)
    print(f'Win rate for Team A: {predicted_win_rate[0][0] * 100:.2f}%')
    print(f'Win rate for Team B: {predicted_win_rate[0][1] * 100:.2f}%')

