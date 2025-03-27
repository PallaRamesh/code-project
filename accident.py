import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np

# Step 1: Import Dataset from GitHub
url = 'https://raw.githubusercontent.com/datasets/traffic-accidents/master/data/traffic-accidents.csv'

# Load dataset directly from GitHub URL
df = pd.read_csv(url)

# Preview the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Step 2: Data Preprocessing
# Convert the 'time' column to datetime (assuming the column is named 'time')
df['time'] = pd.to_datetime(df['time'], errors='coerce')  # If time is not in the correct format

# Extract additional time-related features
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.day_name()

# Handle missing data (if any)
df.dropna(subset=['road_condition', 'weather', 'hour', 'latitude', 'longitude'], inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# 3.1. Accident count by weather condition
weather_accidents = df.groupby('weather')['accident_id'].count().reset_index()
weather_accidents.columns = ['Weather', 'Accident Count']

# 3.2. Accident count by road condition
road_condition_accidents = df.groupby('road_condition')['accident_id'].count().reset_index()
road_condition_accidents.columns = ['Road Condition', 'Accident Count']

# 3.3. Accident count by hour of the day
hourly_accidents = df.groupby('hour')['accident_id'].count().reset_index()
hourly_accidents.columns = ['Hour of Day', 'Accident Count']

# 3.4. Accident count by day of the week
day_of_week_accidents = df.groupby('day_of_week')['accident_id'].count().reset_index()
day_of_week_accidents = day_of_week_accidents.set_index('day_of_week').reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
).reset_index()

# Step 4: Visualization

# 4.1. Accident count by weather condition
plt.figure(figsize=(10, 6))
sns.barplot(x='Weather', y='Accident Count', data=weather_accidents, palette='Set2')
plt.title('Accident Count by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Accident Count')
plt.xticks(rotation=45)
plt.show()

# 4.2. Accident count by road condition
plt.figure(figsize=(10, 6))
sns.barplot(x='Road Condition', y='Accident Count', data=road_condition_accidents, palette='Set1')
plt.title('Accident Count by Road Condition')
plt.xlabel('Road Condition')
plt.ylabel('Accident Count')
plt.xticks(rotation=45)
plt.show()

# 4.3. Accident count by hour of day
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour of Day', y='Accident Count', data=hourly_accidents, marker='o', color='blue')
plt.title('Accident Count by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Accident Count')
plt.xticks(np.arange(0, 24, 1))
plt.show()

# 4.4. Accident count by day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x='day_of_week', y='Accident Count', data=day_of_week_accidents, palette='coolwarm')
plt.title('Accident Count by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Accident Count')
plt.xticks(rotation=45)
plt.show()

# Step 5: Geospatial Analysis - Heatmap of Accident Hotspots
# Create a map centered around the average latitude and longitude of accidents
map_center = [df['latitude'].mean(), df['longitude'].mean()]
traffic_map = folium.Map(location=map_center, zoom_start=12)

# Add heatmap layer
heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(traffic_map)

# Save the map as an HTML file
traffic_map.save('accident_heatmap.html')
print("Heatmap saved as 'accident_heatmap.html'")

# Step 6: Accident Count by Weather and Time of Day (Heatmap)
weather_hourly_accidents = df.groupby(['weather', 'hour'])['accident_id'].count().reset_index()
weather_hourly_accidents.columns = ['Weather', 'Hour of Day', 'Accident Count']
plt.figure(figsize=(12, 6))
pivot = weather_hourly_accidents.pivot('Weather', 'Hour of Day', 'Accident Count')
sns.heatmap(pivot, annot=True, fmt="d", cmap='YlGnBu', linewidths=0.5)
plt.title('Accidents by Weather Condition and Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Weather Condition')
plt.show()
