# AIX_DL.github.io


## [MEMBERS]

성연우 (생명과학과) 2023013118
임종호 (기계공학부) 2024-

# 1. PROPOSAL

   [Motivation: why are you doing this?]

   [What do you want to see at the end?]

# 2. DATASETS

   [Alcohol effects on study]
   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course와 Portuguese language course를 수강하는 학생들의 기본 정보(성별, 거주지, 부모님 직업 등) 및 알코올 섭취 정도, 성적을 포함한다.
   - 두 과목 모두 학생 기본 정보 및 알코올 섭취 정도를 나타내는 30개의 열과 성적을 나타내는 3개의 열로 구성되어있으며, Math course의 경우 395개, Portuguese language course의 경우 649개의 데이터 셋으로 구성되어있다.
   - 데이터 셋은 이진 분류, 5단계 분류, 회귀 방법에 따라 모델링 되었다.

      - 이진분류(binary classification): 입력값에 따라 분류한 카테고리가 두 가지인 분류 알고리즘. 
      - 다중분류(5-level classification): 입력값에 따라 분류한 카테고리가 세 가지 이상(본 데이터셋에서는 다섯 가지)인 분류 알고리즘.
      - 회귀(regression):


  

  [Additional grade and alcohol variables]

3. METHODOLOGY
   - ANN 사용하여 예측 모델 학습. 

```python
import pandas as pd

# 데이터셋 불러오기
math_data = pd.read_csv('/kaggle/input/mathportugeseyouthalcoholstudy/student_math_por_formatted.csv')
study_data = pd.read_csv('/kaggle/input/alcohol-effects-on-study/Maths.csv')

# 데이터셋 확인
print("Math Dataset:")
print(math_data.head())

print("\nStudy Dataset:")
print(study_data.head())
```


```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 결측치 확인 및 제거
math_data = math_data.dropna()
study_data = study_data.dropna()

# Label Encoding (범주형 -> 숫자형)
encoder = LabelEncoder()
for column in ['sex', 'school', 'address']:
    if column in math_data.columns:
        math_data[column] = encoder.fit_transform(math_data[column])

# 스케일링 (필요한 경우)
scaler = StandardScaler()
if 'G1' in math_data.columns:  # 예: G1 점수 스케일링
    math_data[['G1', 'G2', 'G3']] = scaler.fit_transform(math_data[['G1', 'G2', 'G3']])
```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 입력 데이터와 라벨 준비 (예: 'G3'를 타겟으로 사용)
X = math_data.drop(columns=['G3'])
y = math_data['G3']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # 회귀 문제의 경우 활성화 함수 없음
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)
```


```python
import matplotlib.pyplot as plt

# 모델 평가
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```





4. EVALUATION & ANALYSIS
5. RELATED WORK
6. CONCLUSION: DISCUSSION
