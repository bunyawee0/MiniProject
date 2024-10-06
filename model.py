import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout


data = pd.read_csv('data_clean.csv')

# แสดงข้อมูลเบื้องต้น
print(data.head())

# กำหนดฟีเจอร์
features = [
    'Score', 'Pick %', 'Dmg/Round', 'KDA', 
    'Attacker Win %', 'Attacker KDA', 
    'Defender Win %', 'Defender KDA',
    'A Pick %', 'A Defuse %', 'B Pick %', 
    'B Defuse %', 'C Pick %', 'C Defuse %'
]

data['Winner'] = (data['Win %'] > 0.5).astype(int)  # 1 = Team A ชนะ, 0 = Team B ชนะ

# เตรียมข้อมูลสำหรับการฝึกโมเดล
X = data[features].values
y = data['Winner'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape ข้อมูลให้เป็น (samples, 14, 1)
X_train = X_train.reshape((X_train.shape[0], 14, 1))  # (samples, 14, 1)
X_test = X_test.reshape((X_test.shape[0], 14, 1))    # (samples, 14, 1)

# ขั้นตอนที่ 5: สร้างโมเดล Conv1D
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(14, 1)))  # input_shape = (14, 1)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
model.save('cnn_model.h5')

# def predict_winner(agents_a, agents_b, map_name):

#     new_data = []
    
#     for agent in agents_a + agents_b:
#         agent_data = data[data['Agent'] == agent].iloc[0]
#         new_data.append([
#             agent_data['Score'], agent_data['Pick %'], agent_data['Dmg/Round'], agent_data['KDA'],
#             agent_data['Attacker Win %'], agent_data['Attacker KDA'],
#             agent_data['Defender Win %'], agent_data['Defender KDA'],
#             agent_data['A Pick %'], agent_data['A Defuse %'], 
#             agent_data['B Pick %'], agent_data['B Defuse %'],
#             agent_data['C Pick %'], agent_data['C Defuse %']
#         ])
    
#     # แปลงข้อมูลใหม่เป็น DataFrame
#     new_df = pd.DataFrame(new_data, columns=features)

#     # แปลงข้อมูลใหม่เป็นรูปแบบที่เหมาะสมสำหรับการทำนาย
#     # new_X = new_df.values.reshape((1, 14, 1))  # (1, 14, 1)

#     # ทำนายผล
#     prediction = model.predict(new_df)
#     winner = 'Team A' if prediction[0][0] > 0.5 else 'Team B'
#     return winner

# # รับข้อมูลจากผู้ใช้
# agents_team_b = ['Omen', 'Cypher', 'Skye', 'Raze', 'Jett']
# agents_team_a = ['Harbor', 'Killjoy', 'Viper', 'Jett', 'Skye']
# map_name = 'Lotus'

# # agents_team_b = ['Astra', 'Skye', 'Viper', 'Raze', 'Jett']
# # agents_team_a = ['Skye', 'Viper', 'Chamber', 'Astra', 'Raze']
# # map_name = 'Split'

# # ทำนายผู้ชนะ
# predicted_winner = predict_winner(agents_team_a, agents_team_b, map_name)
# print(f'The predicted winner is: {predicted_winner}')
