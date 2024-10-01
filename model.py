import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('data_clean.csv')  # Replace with your CSV file path

print(data.head())

# การเตรียมข้อมูล
# ลบสัญลักษณ์ '%' และแปลงคอลัมน์เป็นประเภทที่เหมาะสม
data['Win %'] = data['Win %'].str.replace('%', '').astype(float) / 100
data['Pick %'] = data['Pick %'].str.replace('%', '').astype(float) / 100
data['Dmg/Round'] = data['Dmg/Round'].astype(float)
data['KDA'] = data['KDA'].astype(float)

# กำหนดฟีเจอร์และเป้าหมาย
features = data[['Score', 'Trend', 'Pick %', 'Dmg/Round', 'KDA']]
features = pd.get_dummies(features, columns=['Trend'], drop_first=True)  # แปลง Trend เป็นตัวเลข
target = data['Win %']

# แบ่งข้อมูลออกเป็นชุดฝึกอบรมและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# ใช้ Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทำนายผลในชุดทดสอบ
y_pred = model.predict(X_test)

# คำนวณความแม่นยำโดยใช้ Mean Squared Error และ R² Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# แสดงกราฟการกระจายค่าที่ทำนายและค่าจริง
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # เส้นที่แสดงถึงค่าจริง
plt.title('Actual vs Predicted Win %')
plt.xlabel('Actual Win %')
plt.ylabel('Predicted Win %')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()

# แสดงความสำคัญของฟีเจอร์
importances = model.feature_importances_
feature_importance = pd.DataFrame(importances, index=features.columns, columns=["Importance"]).sort_values("Importance", ascending=False)

# แสดงกราฟความสำคัญของฟีเจอร์
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid()
plt.show()

joblib.dump(model, 'model.pkl')