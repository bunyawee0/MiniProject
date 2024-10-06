
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def predict_winner(agents_a, agents_b, map_name):
    model = load_model('cnn_model.h5')

    features = [
        'Score', 'Pick %', 'Dmg/Round', 'KDA', 
        'Attacker Win %', 'Attacker KDA', 
        'Defender Win %', 'Defender KDA',
        'A Pick %', 'A Defuse %', 'B Pick %', 
        'B Defuse %', 'C Pick %', 'C Defuse %'
    ]
    new_data = []
    
    for agent in agents_a + agents_b:
        agent_data = data[data['Agent'] == agent].iloc[0]
        new_data.append([
            agent_data['Score'], agent_data['Pick %'], agent_data['Dmg/Round'], agent_data['KDA'],
            agent_data['Attacker Win %'], agent_data['Attacker KDA'],
            agent_data['Defender Win %'], agent_data['Defender KDA'],
            agent_data['A Pick %'], agent_data['A Defuse %'], 
            agent_data['B Pick %'], agent_data['B Defuse %'],
            agent_data['C Pick %'], agent_data['C Defuse %']
        ])
    
    # แปลงข้อมูลใหม่เป็น DataFrame
    new_df = pd.DataFrame(new_data, columns=features)

    # ทำนายผล
    predictions = model.predict(new_df)
    
    # หาค่าเฉลี่ยของค่าทำนายจากตัวที่ 1 ถึงตัวที่ 5
    # average_prediction = np.mean(predictions[:5])

    # กำหนดผู้ชนะจากค่าเฉลี่ย
    # winner = 'Team A' if average_prediction <= 0.5 else 'Team B'
    team_a_mean = predictions[:5].mean()  # Mean of the first 5 values
    team_b_mean = predictions[-5:].mean()  # Mean of the last 5 values

    # Comparison
    if team_a_mean > team_b_mean:
        winner = 'Team A'
        # print("Team A")
    else:
        winner = 'Team B'
        # print("Team B")
        
    # Display the means
    print(f"team a mean: {team_a_mean}")
    print(f"team b mean: {team_b_mean}")
    return winner, predictions, (team_a_mean, team_b_mean)

# รับข้อมูลจากผู้ใช้
data = pd.read_csv('data_clean.csv')
# agents_team_a = ['Omen', 'Cypher', 'Skye', 'Raze', 'Jett']
# agents_team_b = ['Harbor', 'Killjoy', 'Viper', 'Jett', 'Skye']
# map_name = 'Lotus'

agents_team_a = ['Astra', 'Skye', 'Viper', 'Raze', 'Jett']
agents_team_b = ['Skye', 'Viper', 'Chamber', 'Astra', 'Raze']
map_name = 'Split'

# ทำนายผู้ชนะ
# predicted_winner , prediction= predict_winner(agents_team_a, agents_team_b, map_name)
predicted_winner, prediction, avg_prediction = predict_winner(agents_team_a, agents_team_b, map_name)

print(f'The predicted winner is {predicted_winner}')
print(avg_prediction)
# print(prediction)