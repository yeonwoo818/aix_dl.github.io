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

    Math Dataset:
       Unnamed: 0  school  sex  age  address  famsize  Pstatus  Medu  Fedu  \
    0           0       0    0   18        0        1        0     4     4   
    1           1       0    0   17        0        1        1     1     1   
    2           2       0    0   15        0        0        1     1     1   
    3           3       0    0   15        0        1        1     4     2   
    4           4       0    0   16        0        1        1     3     3   
    
       traveltime  ...  reason_course  reason_home  reason_other  \
    0           2  ...              1            0             0   
    1           1  ...              1            0             0   
    2           1  ...              0            0             1   
    3           1  ...              0            1             0   
    4           1  ...              0            1             0   
    
       reason_reputation  guardian_father  guardian_mother  guardian_other  \
    0                  0                0                1               0   
    1                  0                1                0               0   
    2                  0                0                1               0   
    3                  0                0                1               0   
    4                  0                1                0               0   
    
       binge_drinker  heavy_drinker  overall_grade  
    0            0.0            0.0           5.75  
    1            0.0            0.0           5.50  
    2            0.0            0.0           8.75  
    3            0.0            0.0          14.75  
    4            0.0            0.0           9.00  
    
    [5 rows x 51 columns]
    
    Study Dataset:
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  ...  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher  ...   
    1     GP   F   17       U     GT3       T     1     1  at_home     other  ...   
    2     GP   F   15       U     LE3       T     1     1  at_home     other  ...   
    3     GP   F   15       U     GT3       T     4     2   health  services  ...   
    4     GP   F   16       U     GT3       T     3     3    other     other  ...   
    
      famrel freetime  goout  Dalc  Walc health absences  G1  G2  G3  
    0      4        3      4     1     1      3        6   5   6   6  
    1      5        3      3     1     1      3        4   5   5   6  
    2      4        3      2     2     3      3       10   7   8  10  
    3      3        2      2     1     1      5        2  15  14  15  
    4      4        3      2     1     2      5        4   6  10  10  
    
    [5 rows x 33 columns]
    


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

    Epoch 1/50
    

    /opt/conda/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    

    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - loss: 495.4179 - mae: 15.4980 - val_loss: 84.7213 - val_mae: 7.2883
    Epoch 2/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 207.0504 - mae: 9.7510 - val_loss: 30.8200 - val_mae: 4.4037
    Epoch 3/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 137.1797 - mae: 7.8040 - val_loss: 52.8368 - val_mae: 5.8855
    Epoch 4/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 93.0497 - mae: 6.4067 - val_loss: 13.1999 - val_mae: 2.8536
    Epoch 5/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 68.0826 - mae: 5.5576 - val_loss: 19.5678 - val_mae: 3.5202
    Epoch 6/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 47.1431 - mae: 4.6203 - val_loss: 13.5467 - val_mae: 2.9169
    Epoch 7/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 47.3257 - mae: 4.5305 - val_loss: 9.6791 - val_mae: 2.4942
    Epoch 8/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 31.7530 - mae: 3.8355 - val_loss: 5.2769 - val_mae: 1.8271
    Epoch 9/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 21.7891 - mae: 3.2955 - val_loss: 4.9339 - val_mae: 1.7594
    Epoch 10/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 22.2382 - mae: 3.2742 - val_loss: 7.5389 - val_mae: 2.1908
    Epoch 11/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 22.2976 - mae: 2.9823 - val_loss: 3.2362 - val_mae: 1.4383
    Epoch 12/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 18.2404 - mae: 2.8006 - val_loss: 3.7621 - val_mae: 1.5544
    Epoch 13/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 15.4579 - mae: 2.6180 - val_loss: 5.0069 - val_mae: 1.7962
    Epoch 14/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 13.6854 - mae: 2.5447 - val_loss: 4.1591 - val_mae: 1.6391
    Epoch 15/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 13.4332 - mae: 2.4434 - val_loss: 2.3705 - val_mae: 1.2660
    Epoch 16/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 9.1754 - mae: 2.1352 - val_loss: 1.9906 - val_mae: 1.1471
    Epoch 17/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 13.1283 - mae: 2.3571 - val_loss: 1.0373 - val_mae: 0.8202
    Epoch 18/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 10.6930 - mae: 2.2172 - val_loss: 2.6865 - val_mae: 1.3209
    Epoch 19/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 7.4895 - mae: 1.8136 - val_loss: 1.2140 - val_mae: 0.8949
    Epoch 20/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 8.6971 - mae: 1.9072 - val_loss: 0.9415 - val_mae: 0.7348
    Epoch 21/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 6.2355 - mae: 1.7575 - val_loss: 3.6696 - val_mae: 1.5468
    Epoch 22/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 6.4359 - mae: 1.7253 - val_loss: 1.1104 - val_mae: 0.8388
    Epoch 23/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 5.3198 - mae: 1.6087 - val_loss: 1.0145 - val_mae: 0.7933
    Epoch 24/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 4.9710 - mae: 1.5492 - val_loss: 0.9189 - val_mae: 0.7377
    Epoch 25/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 4.6724 - mae: 1.4313 - val_loss: 1.0279 - val_mae: 0.8016
    Epoch 26/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.3341 - mae: 1.2394 - val_loss: 1.7109 - val_mae: 1.0706
    Epoch 27/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 4.3010 - mae: 1.4472 - val_loss: 0.6383 - val_mae: 0.5862
    Epoch 28/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.1200 - mae: 1.2491 - val_loss: 0.8574 - val_mae: 0.7418
    Epoch 29/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.6534 - mae: 1.1237 - val_loss: 0.6324 - val_mae: 0.5462
    Epoch 30/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.8282 - mae: 1.2466 - val_loss: 0.9050 - val_mae: 0.7810
    Epoch 31/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.2537 - mae: 1.1936 - val_loss: 0.5413 - val_mae: 0.5375
    Epoch 32/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.7979 - mae: 1.0962 - val_loss: 0.5600 - val_mae: 0.5930
    Epoch 33/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.3005 - mae: 1.0362 - val_loss: 0.5455 - val_mae: 0.5628
    Epoch 34/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.3762 - mae: 1.0102 - val_loss: 1.1097 - val_mae: 0.9062
    Epoch 35/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.1041 - mae: 1.0477 - val_loss: 0.4594 - val_mae: 0.4169
    Epoch 36/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.6794 - mae: 1.0301 - val_loss: 0.5605 - val_mae: 0.6053
    Epoch 37/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.6679 - mae: 0.8498 - val_loss: 0.5015 - val_mae: 0.5569
    Epoch 38/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.7374 - mae: 0.8765 - val_loss: 0.4021 - val_mae: 0.3929
    Epoch 39/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.6897 - mae: 0.9243 - val_loss: 0.5856 - val_mae: 0.4860
    Epoch 40/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.5849 - mae: 0.8468 - val_loss: 0.4430 - val_mae: 0.4032
    Epoch 41/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.4603 - mae: 0.8782 - val_loss: 0.8155 - val_mae: 0.7757
    Epoch 42/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.3231 - mae: 0.8216 - val_loss: 0.6152 - val_mae: 0.5324
    Epoch 43/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.3276 - mae: 0.8222 - val_loss: 0.7220 - val_mae: 0.6047
    Epoch 44/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.1935 - mae: 0.7336 - val_loss: 0.3519 - val_mae: 0.3654
    Epoch 45/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.2643 - mae: 0.7388 - val_loss: 0.3891 - val_mae: 0.5033
    Epoch 46/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.1903 - mae: 0.6968 - val_loss: 0.3091 - val_mae: 0.3681
    Epoch 47/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.9242 - mae: 0.6599 - val_loss: 0.5228 - val_mae: 0.4737
    Epoch 48/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.9571 - mae: 0.6971 - val_loss: 0.4996 - val_mae: 0.4546
    Epoch 49/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.2589 - mae: 0.7552 - val_loss: 0.6497 - val_mae: 0.5528
    Epoch 50/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.0587 - mae: 0.6736 - val_loss: 0.2717 - val_mae: 0.3143
    


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

    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2593 - mae: 0.2895 
    Mean Absolute Error: 0.29620519280433655
    


    
![png](maths_files/maths_3_1.png)
    



```python
import matplotlib.pyplot as plt

# 모델 훈련 결과 예시 (history 객체에서 가져옴)
history = {
    "loss": [0.8, 0.6, 0.4, 0.2],
    "val_loss": [0.9, 0.7, 0.5, 0.3],
    "accuracy": [0.5, 0.7, 0.85, 0.9],
    "val_accuracy": [0.4, 0.6, 0.8, 0.88]
}

# 손실 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```


    
![png](maths_files/maths_4_0.png)
    



```python
import seaborn as sns
import numpy as np

# 실제값과 예측값 예시
y_true = np.random.randint(0, 2, 100)  # 실제값
y_pred = np.random.random(100)  # 예측 확률값

# 히스토그램으로 분포 시각화
plt.figure(figsize=(8, 6))
sns.histplot(y_pred, kde=True, label='Predicted Probabilities', color='blue')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Predicted Probabilities Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](maths_files/maths_5_1.png)
    



```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 혼동 행렬 생성
cm = confusion_matrix(y_true, y_pred > 0.5)  # 예측값을 0.5 기준으로 이진 분류
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

# 시각화
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
```


    
![png](maths_files/maths_6_0.png)
    



```python
from sklearn.metrics import roc_curve, auc

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# 그래프
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
```


    
![png](maths_files/maths_7_0.png)
    



```python
# 결과 분포 Boxplot
sns.boxplot(x=y_true, y=y_pred)
plt.title('Predicted Values by True Labels')
plt.xlabel('True Labels')
plt.ylabel('Predicted Probabilities')
plt.show()
```


    
![png](maths_files/maths_8_0.png)
    



```python
import pandas as pd

# 피처 중요도 예시 데이터
feature_importance = pd.Series(
    [0.2, 0.3, 0.15, 0.05, 0.1, 0.2], 
    index=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6"]
)

# 막대 그래프
feature_importance.sort_values().plot(kind='barh', color='skyblue', figsize=(8, 6))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
```


    
![png](maths_files/maths_9_0.png)
    

