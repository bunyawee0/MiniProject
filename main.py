# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError

# # โหลดโมเดล พร้อม custom objects
# model = load_model('cnn_model.h5', custom_objects={'mse': MeanSquaredError()})

# def convert_percentage(value):
#     """Convert percentage string to float."""
#     if isinstance(value, str) and '%' in value:
#         return float(value.replace('%', '').strip()) / 100.0  # แปลงเป็นค่า float
#     return float(value) if isinstance(value, (int, float)) else np.nan  # แปลงเป็น float ถ้าเป็น int หรือ float

# def predict_team_winner(training_data, agent_a_list, agent_b_list, map_name):
#     data = []

#     for agent_a, agent_b in zip(agent_a_list, agent_b_list):
#         entry_a = training_data[training_data['Agent'] == agent_a]
#         entry_b = training_data[training_data['Agent'] == agent_b]

#         if not entry_a.empty and not entry_b.empty:
#             row_a = entry_a.iloc[0]
#             row_b = entry_b.iloc[0]

#             # ใช้ Trend ของ Agent ที่ชนะ (หรือเฉลี่ยระหว่างทั้งสอง)
#             trend_value = row_a['Trend'] if row_a['Score'] >= row_b['Score'] else row_b['Trend']
            
#             data.append({
#                 'Score': (row_a['Score'] + row_b['Score']) / 2,
#                 'Pick %': (convert_percentage(row_a['Pick %']) + convert_percentage(row_b['Pick %'])) / 2,
#                 'Dmg/Round': (row_a['Dmg/Round'] + row_b['Dmg/Round']) / 2,
#                 'KDA': (row_a['KDA'] + row_b['KDA']) / 2,
#                 'Trend': trend_value  # ใช้ค่า Trend ที่ถูกต้อง
#             })

#     df_input = pd.DataFrame(data)

#     # แปลง Trend เป็นตัวเลขด้วย one-hot encoding
#     df_input = pd.get_dummies(df_input, columns=['Trend'], drop_first=True)

#     # ตรวจสอบให้แน่ใจว่าฟีเจอร์ทั้งหมดถูกต้อง
#     expected_features = ['Score', 'Pick %', 'Dmg/Round', 'KDA']  # ต้องรวมฟีเจอร์ที่โมเดลคาดหวัง
#     df_input = df_input[expected_features]

#     # ตรวจสอบว่ามีฟีเจอร์ที่ขาด และเติมค่าจาก training_data
#     for feature in expected_features:
#         if df_input[feature].isnull().any() or (df_input[feature] == 0).any():
#             # หาค่าจาก training_data สำหรับ agent ที่ตรงกัน
#             match_row = training_data[training_data['Agent'].isin(agent_a_list + agent_b_list)].mean()  # ใช้ค่าเฉลี่ยของ Agent ที่ตรงกัน
#             if pd.isna(df_input[feature]).all() or (df_input[feature] == 0).all():
#                 # ตรวจสอบประเภทของฟีเจอร์ว่ามีค่าที่ต้องแปลง
#                 if feature == 'Pick %':
#                     df_input[feature] = convert_percentage(match_row[feature])
#                 else:
#                     df_input[feature] = match_row[feature]

#     # เติมฟีเจอร์เพิ่มเติมให้ครบ 28 ตัว
#     while df_input.shape[1] < 28:
#         # เติมค่าจาก training_data
#         extra_feature = training_data[expected_features].mean()  # ค่าเฉลี่ยจาก training_data
#         df_input[f'Extra_Feature_{df_input.shape[1] + 1}'] = extra_feature

#     # ตรวจสอบจำนวนฟีเจอร์ที่โมเดลต้องการ
#     print(f"Input shape before reshaping: {df_input.shape}")

#     # reshape input
#     df_input = df_input.values.reshape((df_input.shape[0], 28, 1))

#     # ตรวจสอบ shape ของ df_input
#     print(f"Input shape after reshaping: {df_input.shape}")

#     # ทำการคาดการณ์
#     predicted_win_rate = model.predict(df_input)

#     team_wins = "Team A Win" if predicted_win_rate.mean() >= 0.5 else "Team B Win"
#     return team_wins, predicted_win_rate

# if __name__ == '__main__':
#     training_data = pd.read_csv('data_clean.csv')

#     # รับข้อมูลจากผู้ใช้
#     agent_a_list = ['Killjoy', 'Brimstone', 'Raze', 'Sova', 'Breach']
#     agent_b_list = ['Breach', 'Killjoy', 'Raze', 'Brimstone', 'Viper']
#     map_name = 'Fracture'

#     predicted_team_winner, predicted_win_rate = predict_team_winner(training_data, agent_a_list, agent_b_list, map_name)

#     print(f'ผลการคาดการณ์ทีมที่ชนะ: {predicted_team_winner}')
#     print(f'Win rate: {predicted_win_rate.mean()*100:.2f}%')

