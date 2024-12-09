```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Step 1: Load the datasets
data1 = pd.read_csv('/kaggle/input/mathportugeseyouthalcoholstudy/student_math_por_formatted.csv')
data2 = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')
# Step 2: Explore the datasets
print("Dataset 1 Shape:", data1.shape)
print("Dataset 2 Shape:", data2.shape)
# Step 3: Merge datasets
# Assume both datasets have similar structure and can be concatenated
common_columns = list(set(data1.columns).intersection(set(data2.columns)))
data = pd.concat([data1[common_columns], data2[common_columns]], axis=0).reset_index(drop=True)
# Handle Inf and NaN values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
print("Merged Dataset Shape:", data.shape)
# Step 4: Preprocess the data
# Ensure all categorical columns have consistent types
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    # Convert mixed types to string
    data[col] = data[col].astype(str)
# Convert categorical columns to numerical values using Label Encoding
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])
# Feature selection (exclude target variable)
features = data.drop(['G3'], axis=1)  # 'G3' is the final grade (target variable)
target = data['G3']
# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Step 7: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
# Step 8: Feature importance
importance = model.feature_importances_
feature_names = features.columns
# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names)
plt.title("Feature Importance")
plt.show()
# Step 9: Data Visualization
# 1. Distribution of Final Grades (G3)
plt.figure(figsize=(8, 6))
sns.histplot(data['G3'], kde=True, bins=20, color='blue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.show()
# 2. Alcohol Consumption vs Final Grades
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Dalc'], y=data['G3'], palette='viridis')
plt.title('Daily Alcohol Consumption vs Final Grades')
plt.xlabel('Daily Alcohol Consumption (Dalc)')
plt.ylabel('Final Grade (G3)')
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Walc'], y=data['G3'], palette='coolwarm')
plt.title('Weekend Alcohol Consumption vs Final Grades')
plt.xlabel('Weekend Alcohol Consumption (Walc)')
plt.ylabel('Final Grade (G3)')
plt.show()
# 3. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()
# Step 10: Save the model (optional)
import joblib
joblib.dump(model, 'student_alcohol_model.pkl')

```

    Dataset 1 Shape: (1044, 51)
    Dataset 2 Shape: (395, 33)
    Merged Dataset Shape: (1439, 29)
    RMSE: 1.2048505843280135
    R^2 Score: 0.9108730198325782
    


    
![png](alcohol-study-and-consumption_files/alcohol-study-and-consumption_0_1.png)
    


    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](alcohol-study-and-consumption_files/alcohol-study-and-consumption_0_3.png)
    



    
![png](alcohol-study-and-consumption_files/alcohol-study-and-consumption_0_4.png)
    



    
![png](alcohol-study-and-consumption_files/alcohol-study-and-consumption_0_5.png)
    



    
![png](alcohol-study-and-consumption_files/alcohol-study-and-consumption_0_6.png)
    





    ['student_alcohol_model.pkl']


