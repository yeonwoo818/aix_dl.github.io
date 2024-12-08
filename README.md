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

   [Student Alcohol Consumption]
   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course와 Portuguese language course를 수강하는 학생들의 기본 정보(성별, 거주지, 부모님 직업 등) 및 알코올 섭취 정도, 성적을 포함한다.
   - 두 과목 모두 학생 기본 정보 및 알코올 섭취 정도를 나타내는 30개의 열과 성적을 나타내는 3개의 열로 구성되어있으며, Math course의 경우 395개, Portuguese language course의 경우 649개의 데이터 셋으로 구성되어있다.
   - 데이터 셋은 이진 분류, 5단계 분류, 회귀 방법에 따라 모델링 되었다.



   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course 와 




      - 이진분류(binary classification): 입력값에 따라 분류한 카테고리가 두 가지인 분류 알고리즘. 
      - 다중분류(5-level classification): 입력값에 따라 분류한 카테고리가 세 가지 이상(본 데이터셋에서는 다섯 가지)인 분류 알고리즘.
      - 회귀(regression):



## III. Methodology
   - Randomforest를 사용하여 학생의 알코올 섭취 및 성적 데이터 분석, 예측
   - 학생의 성적 데이터를 기반으로 예측 모델을 생성.
   - 전처리, 학습, 평가의 전체 머신러닝 파이프라인을 보여주며, 결과적으로 randomforest 모델을 사용하여 학생 성적(G3)을 예측함. 

```python
# Step 1: Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
```
- pandas: 데이터 로드 및 조작에 사용
- numpy: 수치 계산 및 데이터 처리를 위한 라이브러리
- matplotlib.pyplot, seaborn: 데이터 시각화 도구
- sklearn 모듈: 머신러닝 작업을 위해 다양한 모델 및 유틸리티 제공
  - train_test_split: 데이터 학습/테스트 세트로 분할
  - LabelEncoder: 범주형 데이터 숫자로 변환
  - StranderScaler: 데이터 표준화(정규화)
  - RandomForestRegressor: 랜덤 포레스트 회귀 모델
  - mean_squared_error, r2_score: 모델 성능 평가

```python
# Step 2: Load the datasets
data1 = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')
data2 = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')
```
- data1: 학생들의 수학 성적과 관련된 데이터(student-mat.csv)
- data2: 학생들의 포르투갈어 성적과 관련된 데이터(student-por.csv)

```python
# Step 3: Explore the datasets
print("Dataset 1 Shape:", data1.shape)
print("Dataset 2 Shape:", data2.shape)
```
- 데이터의 차원(shape)을 출력하여 데이터셋의 크기를 확인
  - (rows, columns) 형태로 나타남

```python
# Step 4: Merge datasets
# Assume both datasets have similar structure and can be concatenated
common_columns = list(set(data1.columns).intersection(set(data2.columns)))
data = pd.concat([data1[common_columns], data2[common_columns]], axis=0).reset_index(drop=True)
# Handle Inf and NaN values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
print("Merged Dataset Shape:", data.shape)
```
- 두 데이터셋에서 공통 column만 사용하여 데이터 병합
  - set.intersection: 공통 column 찾기
  - pd.concat: 두 데이터프레임을 세로로 이어붙임
- 병합 후, 무한값(Inf)과 결측값(NaN) 제거

- 두 데이터셋에서 common_colums만 사용해 데이터를 병합하는 이유: 데이터의 구조적 일관성을 유지하기 위해.
  - 두 데이터셋이 서로 다른 column 구조를 가질 수 있음. 공통 column만 선택하면 병합 후에도 데이터의 각 열의 의미가 일관되게 유지됨.
  - 예를 들어, 두 데이터셋에 모두 존재하는 column이 학생 이름, 나이, 성별 등이라면 이 정보만 병합에 사용됨.
- 결측값 및 불필요한 정보 제거
  - 공통 column이 아닌 데이터는 의미가 없거나 분석에 방해될 수 있음.  

```python
# Step 5: Preprocess the data
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
- 범주형 데이터를 식별하고, 모든 값을 문자열로 변환
- Label Encoding: 범주형 데이터를 숫자로 변환

- features와 target 분리
- G3: 최종 성적(예측 목표) / 나머지 column: 특징(설명 변수)로 사용

- StandardScaler: 모든 특징값을 평균 0, 표준편차 1로 변환
  - 모델 학습 효율을 높이고 과적합 방지

```python
# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
```
- Train-Test Split: 데이터를 80% 학습, 20% 테스트로 분리
  - random_state = 42: 재현 가능한 결과를 보장

```python
# Step 7: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- Random Forest Regressor:
  - 랜덤하게 선택된 여러 개의 트리(Decision Trees)로 구성된 회귀 모델
  - n_estimators = 100: 트리의 개수 설정

```python
# Step 8: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
```
- y_pred: 테스트 데이터를 예측한 결과
- 평가지표:
  - RMSE: 예측값과 실제값의 평균 오차(낮을수록 좋음)
  - R^2 Score: 모델 설명력(1에 가까울수록 좋음)
 
----------------


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 데이터셋 로드
math_df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')
por_df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')

# 알코올 소비 데이터 (수준별 빈도 데이터)
alcohol_levels = {
    "low level": {"Frequency": 1054, "Percentage": 55.9},
    "hazardous level": {"Frequency": 679, "Percentage": 36.0},
    "harmful level": {"Frequency": 154, "Percentage": 8.2}
}

# 2. 수학 및 포르투갈어 데이터셋 병합
students_df = pd.concat([math_df, por_df], ignore_index=True)

# 3. 알코올 소비 수준 분류
def classify_alcohol_level(dalc, walc):
    avg_alcohol = (dalc + walc) / 2
    if avg_alcohol <= 2.0:
        return "low level"
    elif avg_alcohol <= 3.5:
        return "hazardous level"
    else:
        return "harmful level"

students_df['alcohol_level'] = students_df.apply(lambda x: classify_alcohol_level(x['Dalc'], x['Walc']), axis=1)

# 4. 범주형 변수 인코딩
label_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
label_encoder = LabelEncoder()

for col in label_columns:
    students_df[col] = label_encoder.fit_transform(students_df[col])

# 5. 수치형 변수 정규화
numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
scaler = MinMaxScaler()

students_df[numerical_columns] = scaler.fit_transform(students_df[numerical_columns])

# 6. 특성과 타겟 설정
x = students_df.drop(columns=['G3', 'alcohol_level'])  # G3는 최종 성적, alcohol_level은 보조 정보
y = students_df['G3']  # 타겟 변수 (최종 성적)

# 7. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 8. 딥러닝 모델 생성
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # 회귀 문제이므로 선형 활성화 함수 사용

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 9. 모델 학습
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 10. 모델 평가
loss, mae = model.evaluate(x_test, y_test)
print(f'테스트 세트에서의 평균 절대 오차 (MAE): {mae}')
```





## IV. Evaluation & Analysis
## V. Related Work 
## VI. Conclusion: Discussion
