# 알코올 섭취량에 따른 학업성취도 예측
- 한 줄 설명
- youtube link
  
## Members
성연우 | 생명과학과 1학년 | syeonu818@gmail.com

임종호 | 기계공학부 1학년 | oscar0330@hanyang.ac.kr

## I. Proposal
   ### Motivation: why are you doing this? ###
   
   ### What do you want to see at the end? ###

## II. Datasets

   [Additional grade and alcohol variables]
   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course와 Portuguese language course를 수강하는 학생들의 기본 정보(성별, 거주지, 부모님 직업 등) 및 알코올 섭취 정도, 성적을 포함한다.
   - 두 과목 모두 학생 기본 정보 및 알코올 섭취 정도를 나타내는 30개의 열과 성적을 나타내는 3개의 열로 구성되어있으며, Math course의 경우 395개, Portuguese language course의 경우 649개의 데이터 셋으로 구성되어있다.
   - 데이터 셋은 이진 분류, 5단계 분류, 회귀 방법에 따라 모델링 되었다.



   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course 와 




      - 이진분류(binary classification): 입력값에 따라 분류한 카테고리가 두 가지인 분류 알고리즘. 
      - 다중분류(5-level classification): 입력값에 따라 분류한 카테고리가 세 가지 이상(본 데이터셋에서는 다섯 가지)인 분류 알고리즘.
      - 회귀(regression):


  

  [Additional grade and alcohol variables]


## III. Methodology
   - ANN 사용하여 예측 모델 학습.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
```

```python
# Step 1: Load the datasets
data1 = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')
data2 = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')
```

```python
# Step 2: Explore the datasets
print("Dataset 1 Shape:", data1.shape)
print("Dataset 2 Shape:", data2.shape)
```

```python
# Step 3: Merge datasets
# Assume both datasets have similar structure and can be concatenated
common_columns = list(set(data1.columns).intersection(set(data2.columns)))
data = pd.concat([data1[common_columns], data2[common_columns]], axis=0).reset_index(drop=True)
# Handle Inf and NaN values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
print("Merged Dataset Shape:", data.shape)
```

```python
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
features = data.drop(['G3'], axis=1) # 'G3' is the final grade (target variable)
target = data['G3']
# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

```python
# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
```

```python
# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

```python
# Step 7: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
```




## IV. Evaluation & Analysis
## V. Related Work 
## VI. Conclusion: Discussion
