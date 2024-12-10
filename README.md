# 알코올 섭취량에 따른 학업성취도 예측
- youtube link
  
## Members
성연우 | 생명과학과 1학년 | syeonu818@gmail.com

임종호 | 기계공학부 1학년 | oscar0330@hanyang.ac.kr

## I. Proposal
   - Motivation : 
     - 서로 만난 팀원이 1학년이었다. 아직 대학 생활 경험이 적고 적응하는 단계이기에 음주와 시험공부를 번갈아 하는 것을 경험하였다. 이런 상황에서 알코올 섭취와 성적에 관한 데이터셋을 발견하게 되었고, 가장 공감이 되는 주제라 생각하여 알코올 섭취량에 따른 학업성취도를 예측하는 모델을 만들게 되었다. 
   
   - What do you want to see at the end? 
     - 알코올의 섭취와 성적의 상관관계를 알아냄으로써 음주를 주로 하게 되는 시기엔 어떤 과목에 대해 학습을 하는 것 유리한지 살펴보고, 학습에 조금은 더 유리한 알코올의 섭취 시기를 알아내어 학습에 조금은 더 유리한 음주 생활을 찾아보고자 한다.

## II. Datasets

총 두 가지 데이터셋을 활용하였다. 

   ### 1. Student Alcohol Consumption
   - 학생 정보와 알코올 섭취량, 이에 따른 학업 성취도를 일일이 수집하기에는 한계가 있으므로 kaggle에서 제공하는 데이터셋을 사용하였다. (https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
     
   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course와 Portuguese language course를 수강하는 학생들의 기본 정보들과 알코올 섭취 정도, 3개 학기 성적을 포함한다.
     
   - 두 과목 모두 학생의 정보를 나타내는 30개의 열과 성적을 나타내는 3개의 열로 구성되어있으며, Math course의 경우 395개, Portuguese language course의 경우 649개의 데이터셋으로 구성되어있다.

    
  ### 2. Frequency & percentage of alcohol consumption of Austrailian university students

- The association between levels of alcohol consumption and mental health problems and academic performance among young university students (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178142) 에 있는 **alcohol consumption level (학생별 알코올 섭취 수준)** 데이터를 사용하였다.

- 2518명의 학생의 알코올 섭취 수준에 대한 빈도(frequency)와 백분율(percentage)를 나타내는 데이터이며, 개별 학생 데이터는 포함하지 않는다.

--------

- 이 두 데이터를 결합하기 위해 논문 데이터셋(두번째 데이터셋)을 학생별 알코올 섭취 수준을 분류하는 기준으로 설정하고, kaggle 데이터셋에 학생별 알코올 섭취 수준을 할당하였다. 
  
- kaggle 데이터셋에서 Dalc 와 Walc는 각각 주중과 주말에 섭취한 알코올 양을 의미한다. 두 값의 평균을 구하여 그 값이 속하는 범위에 따라 알코올 섭취 정도를 low-level, hazardous-level, harmful-level 로 정의하였다.
  
- kaggle 데이터셋에서 알코올 섭취 정도를 1(아주 낮음)에서 5(아주 높음)까지 정수로 표현하였다.

- 이에 따라 low-level / hazardous-level / harmful-level의 범위는 각각 1.0 ~ 2.33 / 2.34 ~ 3.67 / 3.68 ~ 5.0 으로, 1.0 에서 5.0 까지의 범위를 3등분하여 설정하였다. 여기서 숫자를 이해하기 쉽도록 반올림하여 1.0 ~ 2.0 / 2.1 ~ 3.5 / 3.6 ~ 5.0 으로 설정하였다.
  
  - 위와 같이 <ins>(1) 단순히 범위 기반</ins>으로 설정할 수 있고 이외에도 <ins>(2) 분포에 기반</ins>하여 경계값을 설정하거나 <ins>(3) 분위수에 기반</ins>하여 경계값을 설정할 수 있다. 단순 범위 기반의 경우 빠르고 간단한 분석에 적합하며 현재 상황에서 충분히 유용하다고 판단하여 단순 범위 기반으로 경계값을 정하였다. 





## III. Methodology
   - 학생의 개인적, 가정적, 행동적 요인, 학업 데이터, 알코올 소비 수준 데이터를 바탕으로 학생의 최종 성적을 예측하는 회귀 모델을 구축한다.
   - 딥러닝 기반의 다층 퍼셉트론(Multi-Layer Perceptron)모델을 사용한다.


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

- pandas: 데이터 조작 및 분석을 위한 라이브러리
          데이터프레임 형식으로 데이터를 불러오고, 정리하여 분석하는데 사용함.

- numpy: 수치 계산 및 데이터 처리를 위한 라이브러리
         다양한 수학 함수를 제공. 수치 연산, 배열 및 행렬 연산에 사용함.

- sklearn 모듈: 머신러닝 작업을 위해 다양한 모델 및 유틸리티 제공
  
  - train_test_split: 데이터를 학습 세트와 테스트 세트로 나누는 함수.
                      모델 학습 및 평가를 위해 데이터를 분리함.
  
  - LabelEncoder: 카테고리형 데이터를 숫자로 변환하는 도구.
                  텍스트 기반의 범주형 데이터를 머신러닝 모델에 입력할 수 있는 숫자형 데이터로 변환.
  
  - MinMaxScaler: 데이터의 값을 특정 범위로(기본적으로 0~1)로 스케일링함.
                  feature scaling을 통해 모델 학습을 안정화시킴.

    * feature scaling: 데이터의 각 특성(feature)이 서로 다른 단위를 가질 때, 이를 동일한 범위로 변환하여 머신러닝 알고리즘의 성능을 향상시키는 과정
    
- TensorFlow 모듈: 오픈 소스 머신러닝 및 딥러닝 라이브러리.
  
- Keras 모듈: 딥러닝 모델을 쉽게 구축하고 훈련할 수 있도록 설계된 딥러닝 라이브러리. TensorFlow 같은 딥러닝 프레임워크 위에서 동작함.
  
  - Sequential: 계층(layer)을 순차적으로 쌓아 신경망 모델을 생성.
                딥러닝 모델을 정의함.
  
  - Dense: 완전 연결 계층(Fully Connected Layer).
           신경망의 기본 계층으로, 각 노드가 이전 계층의 모든 노드와 연결됨.
    
  - Dropout: 과적합(overfitting)을 방지하기 위해 일부 뉴런을 무작위로 비활성화하는 계층.
             모델의 일반화 성능을 향상시키는데 사용.

```python
# 1. 데이터셋 로드
math_df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')
por_df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')
```
- 두 csv 파일(student-mat.csv, student-por.csv)을 pandas 데이터프레임으로 로드한다.
  - math_df: 수학 과목 학생 데이터
  - por_df: 포르투갈어 과목 학생 데이터

```python
# 알코올 소비 데이터 (수준별 빈도 데이터)
alcohol_levels = {
    "low level": {"Frequency": 1054, "Percentage": 55.9},
    "hazardous level": {"Frequency": 679, "Percentage": 36.0},
    "harmful level": {"Frequency": 154, "Percentage": 8.2}
}
```
- 논문 데이터의 알코올 섭취 수준 별 빈도와 백분율 값이다. 
```python
# 2. 수학 및 포르투갈어 데이터셋 병합
students_df = pd.concat([math_df, por_df], ignore_index=True)
```
- 두 데이터프레임(math_df, por_df)을 하나로 결합한다.
- ignore_index=True를 통해 새로운 데이터프레임의 인덱스를 재설정한다.
- ignore_index=True는 pandas의 pd.contact에서 사용되는 매개변수로, 데이터를 병합할 때 기존의 인덱스를 무시하고 새로 연속적인 인덱스를 생성하도록 지정한다.
  - ignore_index=False를 하면 병합할 때 원래 데이터프레임의 인덱스를 유지한다.

```python
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
```
- Dalc(주중 음주 수준)과 Walc(주말 음주 수준)의 평균을 구하여 세 가지 음주 수준으로 분류한다.
- apply를 사용하여 각 행에 classify_alcohol_level 함수를 적용한다.
  - apply: pandas에서 제공하는 함수로, 데이터프레임의 각 행 또는 열에 함수를 적용할 때 사용한다. 위 경우 axis = 1 이므로 행에 함수를 적용한다.
  
```python
# 4. 범주형 변수 인코딩
label_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
label_encoder = LabelEncoder()

for col in label_columns:
    students_df[col] = label_encoder.fit_transform(students_df[col])
```
- 학생 데이터의 범주형 변수(학교 이름, 성별, 거주지 유형 등)을 숫자로 변환한다.
- LabelEncoder을 사용해 각 범주형 값을 정수로 값을 매긴다.

```python
# 5. 수치형 변수 정규화
numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
scaler = MinMaxScaler()

students_df[numerical_columns] = scaler.fit_transform(students_df[numerical_columns])
```
- 수치형 데이터(나이, 통학 시간, 학습 시간 등)을 0과 1 사이의 값으로 정규화하여 모델 학습에 적합한 형태로 변환한다.
- MinMaxScaler를 사용하여 값이 최소 0, 최대 1 사이에 위치하도록 스케일링한다.

```python
# 6. 특성과 타겟 설정
x = students_df.drop(columns=['G3', 'alcohol_level'])  # G3는 최종 성적, alcohol_level은 보조 정보
y = students_df['G3']  # 타겟 변수 (최종 성적)
```
- 특성(x): G3(최종성적)과 alcohol_level(음주 수준)을 제외한 나머지 열을 특성을 나타내는 변수로 설정한다.
- 타겟(y): G3 열을 예측 대상 변수로 설정한다.

```python
# 7. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- 데이터셋을 학습 세트(80%)와 테스트 세트(20%)로 분할한다.
- random_state는 데이터 분할 시 난수를 생성하는 초기값(시드)을 고정한다. 시드가 고정되면 항상 같은 방식으로 데이터를 분할하므로 동일한 결과를 나타낸다. 
  - random_state = 42 (임의의 값)을 통해 동일한 결과를 재현 가능하도록 하여 신뢰성을 높인다.

```python
# 8. 딥러닝 모델 생성
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # 회귀 문제이므로 선형 활성화 함수 사용

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```

- Dense: Fully connected layer(완전 연결 층)을 구현하는 Keras의 레이어 클래스이다.
  - 입력과 출력을 각각의 퍼셉트론이 완전 연결하는 신경망의 기본 구성 요소이다.
  - Dense의 주요 매개변수:
    - <ins>units</ins>: layer에서 생성할 퍼셉트론의 개수(128, 64, 32)이다. 
    - <ins>activation</ins>: 활성화 함수. 퍼셉트론의 출력 값을 비선형 변환할 때 사용한다.
      - ReLU(Rectified Linear Unit) 함수는 음수 값을 0으로 바꾸고 양수는 그대로 유지한다. 비선형 특성을 추가하여 모델이 복잡한 패턴을 학습할 수 있게 한다.
    - <ins>input_dim</ins>: 입력 데이터의 차원이다. 
        - input_dim=x_train.shape[1]: 첫 layer에서 입력 데이터의 특성 수를 명시적으로 정의한 것이다.


-  딥러닝 모델을 (input layer / hidden layer / output layer) 정의한다.
  - <ins>input layer</ins>: Dense(128)은 특성 개수(input_dim)에 따라 첫 번째 hidden layer을 생성한다.
  - <ins>hidden layer</ins>:
      - 128 → 64 → 32 perceptron을 가진 3개의 hidden layer을 형성한다.
      - ReLU 함수를 사용한다.
      - Dropout(20%): 과적합 방지를 위해 perceptron의 일부를 무작위로 비활성화해 특정 퍼셉트론의 과도한 의존을 줄인다. 각 훈련 단계에서 퍼셉트론의 20%를 무작위로 비활성화 하는 것을 의미한다.
  - <ins>output layer</ins>: Dense(1)은 단일 연속형 값을 출력한다.

- compile
  - adam: 적응형 학습률 최적화 알고리즘
  - mean_squared_error: 손실 함수로 평균 제곱 오차를 사용한다.
  - mae: 모델 성능 측정 지표로 평균 절대 오차를 사용한다.
 
```python
# 9. 모델 학습
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```
- epochs=50: 50번 반복 학습을 의미한다.
- batch_size=32: 각 batch에 32개의 샘플을 사용한다.
- validation_split=0.2: 학습 데이터 중 20%를 검증 데이터로 사용한다.

```python
# 10. 모델 평가
loss, mae = model.evaluate(x_test, y_test)
print(f'테스트 세트에서의 평균 절대 오차 (MAE): {mae}')
```
- 테스트 데이터를 사용해 모델을 평가한다.
- 평균 절대 오차(mae)를 출력해 예측값과 실제값의 평균 차이를 확인한다.
  - MAE는 모델이 예측에서 얼마나 벗어나는지를 나타낸다.




## IV. Evaluation & Analysis

```python
# 학습 과정 시각화
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 학습과 검증 손실 그래프
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 학습과 검증 MAE 그래프
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
```
위 코드를 통해 학습과정을 시각화하는 그래프, 학습과 검증 손실 그래프, 학습과 검증 MAE그래프를 생성할 수 있다. 

![Training and Validation Loss/MAE table1](./blog-code_files/blog-code_1_1.png)

위 그래프는 학습과 검증 손실 그래프이다. 이 그래프는 훈련 손실과 검증 손실이 함께 수렴하며 감소할 때 모델이 데이터를 잘 학습하고 있음을 나타낸다.
하지만 훈련손실은 감소하나, 검증 손실은 감소하지 않는 추세를 보인다. 이를 통해 과적합을 의심해볼 수 있다. 

과적합을 해결하는 방법으로 더 많은 데이터를 확보하거나 모델을 간소화하는 방법이 있다. 

![Training and Validation Loss/MAE table2](./blog-code_files/blog-code_1_2.png)

위 그래프는 평균의 절대오차(MAE)의 그래프이다. MAE 그래프는 모델의 예측이 실제 값과 얼마나 가까운지 나타낸다.
훈련 mae과 검증 mae 두 값이 비슷하면 모델이 균형 잡힌 학습을 하고 있음을 나타낸다. 하지만 차이가 크다면 과적합 가능성이 있다. 위 그래프도 마찬가지로 그래프가 크게 흔들리므로 학습 과정이 불안정하거나 데이터가 불균형할 가능성이 있다고 판단된다.

이 과적합을 해결하는 방법은 위와 마찬가지로 더 많은 데이터를 확보하거나 모델을 간소화하면 과적합을 해결할 수 있다.
```python
# 예측 값 계산
y_pred = model.predict(x_test)

# 실제 값과 예측 값 비교
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([0, 1], [0, 1], '--', color='red', transform=plt.gca().transAxes)  # y=x 선
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual G3 Scores')
plt.ylabel('Predicted G3 Scores')
plt.show()
```
위 코드를 통해 실제값과 예측값을 비교하는 그래프를 생성할 수 있다.
![Predicted vs Actual table1](./blog-code_files/blog-code_2_1.png)

위 그래프는 예측값과 실제값을 그래프로 나타낸다. 빨간색 실선은 실제값이며, 파란색 값은 예측값을 나타낸다.
빨간색 실선에 파란색 값이 밀집되게 분포되어 있을 수록 모델의 예측 정도를 신뢰할 수 있다는 것을 의미한다.

위 그래프에 경우 G3 Grade가 0.0, 7.5일 때를 제외하고 심하게 산재되어 있지 않다. 따라서 이런 특정 값을 제외하면 모델의 예측 정도를 어느정도 신뢰할 수 있다.

```python
# 잔차 계산
residuals = y_test - y_pred.flatten()

# 히스토그램으로 잔차 분석
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 잔차 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='green')
plt.axhline(0, linestyle='--', color='red')
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
```
위 코드를 통해 잔차를 비교할 수 있는 그래프를 생성할 수 있다. 이 때 진차는 실제값 - 예측값을 나타내며, 모델의 예측 값이 실제 값에서 얼마나 벗어났는지를 알아낼 수 있다. 위 코드는 잔차의 빈도 그래프와 잔차의 산점도 그래프를 생성한다.

![Residual Analysis table1](./blog-code_files/blog-code_3_1.png)

위 그래프는 차가 나오는 빈도를 나타내는 그래프이다. 그래프 내에서 잔차값이 낮은 빈도가 높을수록 모델의 예측이 정확하다.

위 그래프를 보면 잔차가 0과 2 사이일 때 매우 잦은 빈도가 발생한 것을 나타내고 있다. 위 잔차 정의를 보았을 때 이 모델은 예측값과 실제값이 유사한 빈도가 높다는 것을 나태내었으므로 모델의 예측 정도를 신뢰할만 하다고 볼 수 있다.

![Residual Analysis table2](./blog-code_files/blog-code_3_2.png)

위 그래프는 잔차 산점도 그래프를 나타낸다. 잔차 산점도는 모델의 예측 값과 잔차를 비교하여, 모델이 데이터를 잘 적합하고 있는지 확인하는 데 사용된다. 이 그래프가 나타내는 잔차의 패턴과 분포를 해석하는 방법으론 대표적으로 4가지가 있다.

1.잔차가 패턴이 없는 무작위 분포할 때
모델이 데이터를 잘 적합했다는 것을 의미한다.
예측값에 따른 잔차의 분포가 일정하다는 것을 의미한다.

2. 잔차가 곡선
모델이 비선형성을 제대로 학습하지 못했을 가능성이 있다.
이에 대한 예시로 포물선에 선형 회귀 모델을 적용했을 때를 예로 들 수 있다.

3. 팬 모양
모델이 특정 구간에서 더 큰 오차를 발생했다는 것을 의미한다.
데이터 변환 또는 특정 재설계가 필요하다.

4. 잔차가 한쪽으로 치우침
  모델이 편향된 예측을 하고 있을 가능성이 있다.
모델 구조 재설계 또는 추가 데이터가 필요하다.

위 그래프는 잔차가 붉은 색 기준으로 양의 영역으로 치우친 것을 나타내므로 모델이 편향된 예측을 하고 있을 가능성이 있다.
이를 해결하기 위해선 모델 구조를 재설계하거나 추가 데이터가 필요하다.
```python
# 랜덤 포레스트로 특성 중요도 계산
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
importances = rf_model.feature_importances_

# 특성 중요도 시각화
features = x.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
```
위 코드를 통해 변수의 특성 중요도를 파악할 수 있는 그래프를 생성할 수 있다.
![Peature Importance table](./blog-code_files/blog-code_4_0.png)

위 그래프는 특성 중요도를 나타내는 그래프다. 이 그래프를 통해 각 변수의 상대적 중요도를 알 수 있다. 

위 그래프에 따르면 G2의 변수가 압도적으로 중요하다는 것을 알 수 있다. 이 다음으론 결석의 수가 약간 중요하다는 것을 알 수 있다.


## V. Related Work 
- dataset
    - Student Alcohol Consumption : https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
    - Frequency & percentage of alcohol consumption of Austrailian university students : https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178142
- Execution Platform
    - Kaggle : https://www.kaggle.com/
           

## VI. Conclusion: Discussion

1번 그래프(training and validation loss)의 경우 validation loss 그래프와 training loss 가 수렴하며 우하향하는 형태를 띄어야 하는데 validation loss 가 1. training loss랑 수렴하지 않고 2. 우하향하지 않고 증감을 반복하는 추세를 보인다. 

2번 그래프(training and validation MAE)의 경우 validation loss 그래프의 흔들리는 정도가 큰 것을 보아 과적합이 의심됨. 모델이 균형잡힌 학습을 하고 있지 않다는 것. 

3번 그래프는 actual G3 점수가 0.0, 7.5 일 때의 값을 제외하면 나머지 점수 대에서의 값들은 예측값에 근접한 결과를 보임. 

4번 그래프는 3번 그래프의 잔차 빈도를 나타낸 값이므로 3번과 동일한 결과임. 즉, 성적이 0.0, 7.5인 경우를 제외하면 낮은 잔차값(0에 가까운) 빈도가 높으므로 실제값과 어울리는 예측을 하고 있다고 볼 수 있다. 

5번 그래프는 잔차 산점도 그래프이다. 제대로 학습이 이루어졌다면 점들이 패턴 없이 무작위로 분포해야한다. 그러나 본 그래프는 잔차의 양의 영역으로 치우쳤으므로 모델이 편향된 예측을 하고 있음을 알 수 있다. 

6번 그래프는 특성 중요도 그래프이다. G2 의 중요도를 과하게 높게 평가하는 것으로 보아 모델이 편향된 예측을 하고 있음을 알 수 있다. 

위 해석들을 바탕으로 해당 학습 모델은 편향된 예측을 하는 모델임을 알 수 있다. 

이에 대한 원인을 찾아보자면 데이터가 편향 되어 모델이 특정 패턴을 과도하게 학습하여 과적합으로 이어졌을 수 있다. 특히나 특성 중요도 그래프를 살펴보면 G2의 데이터를 과도하게 학습한 것을 볼 수 있다.
이 모델이 이로 인해 과적합이 되어 특정 데이터의 편향이나 노이즈를 반영하는 편향된 예측을 한 것으로 생각된다.

이 문제를 해결하기 위해선 앞서 말했듯이 더 많은 데이터를 확보하거나 모델을 간소화하면 과적합을 해결할 수 있다.


편향된 데이터 → 과적합
편향된 데이터를 학습하게 되면, 모델이 데이터의 특정 패턴(올바르지 않은 패턴 포함)을 과도하게 학습하여 과적합으로 이어질 수 있습니다.
예: 학습 데이터에서 특정 클래스가 과도하게 많으면, 모델이 해당 클래스의 예측 확률을 지나치게 높게 설정할 가능성이 큽니다. 이는 학습 데이터에서 잘 맞는 것처럼 보이지만, 새로운 데이터에서 실패하게 됩니다.
(2) 과적합 → 편향된 예측
모델이 학습 데이터에 과적합되면, 그 데이터의 특정한 편향이나 노이즈를 반영하는 편향된 예측을 할 수 있습니다.
예: 학습 데이터의 특정 속성(예: 성별, 나이 등)과 결과 사이의 관계가 실제와 다름에도 불구하고, 모델이 이를 일반적인 패턴으로 학습하면 예측 결과가 편향될 수 있습니다.




