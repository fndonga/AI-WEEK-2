# ===========================
# SDG 13: Climate Action
# Predict Monthly CO2 Emissions Using Supervised Learning
# ===========================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===========================
# Step 2: Simulate Dataset
# In a real project, replace this with real-world data
# Columns: 'Month', 'Temperature', 'Industry_Index', 'Vehicle_Count', 'CO2_Emissions'
# ===========================

np.random.seed(42)  # For reproducibility
months = np.arange(1, 13)  # Months 1-12
data = []

for month in months:
    for i in range(10):  # Simulate 10 cities/records per month
        temperature = np.random.normal(25, 5)  # Average temp in Celsius
        industry_index = np.random.uniform(50, 150)  # Industrial activity index
        vehicle_count = np.random.randint(5000, 50000)  # Number of vehicles
        # CO2 emissions depend on all three features with some noise
        co2 = 0.5*temperature + 0.3*industry_index + 0.2*vehicle_count/1000 + np.random.normal(0, 5)
        data.append([month, temperature, industry_index, vehicle_count, co2])

# Create DataFrame
df = pd.DataFrame(data, columns=['Month', 'Temperature', 'Industry_Index', 'Vehicle_Count', 'CO2_Emissions'])
print("Sample Data:")
print(df.head())

# ===========================
# Step 3: Visualize Data
# ===========================
sns.pairplot(df, x_vars=['Temperature', 'Industry_Index', 'Vehicle_Count'], y_vars='CO2_Emissions', height=4, kind='scatter')
plt.suptitle("Feature vs CO2 Emissions", y=1.02)
plt.show()

# ===========================
# Step 4: Preprocess Data
# ===========================
# Features and target
X = df[['Month', 'Temperature', 'Industry_Index', 'Vehicle_Count']]
y = df['CO2_Emissions']

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# Step 5: Train Model
# ===========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===========================
# Step 6: Evaluate Model
# ===========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# ===========================
# Step 7: Visualize Predictions
# ===========================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()

# ===========================
# Step 8: Ethical Reflection (Comment)
# ===========================
# 1. Bias: Simulated data may not reflect real-world disparities between cities or industries.
# 2. Fairness: A real model should consider environmental justice (e.g., vulnerable populations).
# 3. Sustainability: The model can help governments or organizations plan interventions to reduce CO2 emissions.
