import pandas as pd

# โหลดไฟล์ CSV ทั้งสามไฟล์
df1 = pd.read_csv('all_maps_ranks_combined.csv')
df2 = pd.read_csv('map_allranks_combined.csv')
df3 = pd.read_csv('plant_allranks_combined.csv')

# ถ้าคุณต้องการรวมไฟล์โดยเชื่อมโยงด้วยคอลัมน์ เช่น 'Map' (สมมติว่าเป็นคอลัมน์ร่วมกัน)
combined_df = pd.merge(df1, df2, on='Map')
combined_df = pd.merge(combined_df, df3, on='Map')

# ถ้าคุณต้องการรวมแถวเข้าด้วยกัน
# combined_df = pd.concat([df1, df2, df3], axis=0)

# แสดงข้อมูล
print(combined_df.head())
combined_df.to_csv('data.csv', index=False)