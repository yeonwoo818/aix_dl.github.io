```python
import pandas as pd

# 데이터셋 불러오기
por_data = pd.read_csv('/kaggle/input/mathportugeseyouthalcoholstudy/student_math_por_formatted.csv')
study_data = pd.read_csv('/kaggle/input/alcohol-effects-on-study/Portuguese.csv')

# 데이터셋 확인
print("por Dataset:")
print(por_data.head())

print("\nStudy Dataset:")
print(study_data.head())
```

    por Dataset:
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
    0      4        3      4     1     1      3        4   0  11  11  
    1      5        3      3     1     1      3        2   9  11  11  
    2      4        3      2     2     3      3        6  12  13  12  
    3      3        2      2     1     1      5        0  14  14  14  
    4      4        3      2     1     2      5        0  11  13  13  
    
    [5 rows x 33 columns]
    


```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 결측치 확인 및 제거
por_data = por_data.dropna()
study_data = study_data.dropna()

# Label Encoding (범주형 -> 숫자형)
encoder = LabelEncoder()
for column in ['sex', 'school', 'address']:
    if column in por_data.columns:
        por_data[column] = encoder.fit_transform(por_data[column])

# 스케일링 (필요한 경우)
scaler = StandardScaler()
if 'G1' in por_data.columns:  # 예: G1 점수 스케일링
    por_data[['G1', 'G2', 'G3']] = scaler.fit_transform(por_data[['G1', 'G2', 'G3']])
```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 입력 데이터와 라벨 준비 (예: 'G3'를 타겟으로 사용)
X = por_data.drop(columns=['G3'])
y = por_data['G3']

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
    

    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 11ms/step - loss: 899.0738 - mae: 20.3578 - val_loss: 130.8967 - val_mae: 9.4605
    Epoch 2/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 193.9590 - mae: 9.1242 - val_loss: 1.5037 - val_mae: 0.9515
    Epoch 3/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 88.3293 - mae: 5.8674 - val_loss: 13.9335 - val_mae: 2.8117
    Epoch 4/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 59.3096 - mae: 5.0592 - val_loss: 5.3268 - val_mae: 1.5984
    Epoch 5/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 46.9715 - mae: 4.6615 - val_loss: 10.7808 - val_mae: 2.4637
    Epoch 6/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 33.8130 - mae: 3.9208 - val_loss: 6.0555 - val_mae: 1.7736
    Epoch 7/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 23.3813 - mae: 3.2605 - val_loss: 10.2857 - val_mae: 2.4921
    Epoch 8/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 16.9316 - mae: 2.8842 - val_loss: 2.3818 - val_mae: 1.0500
    Epoch 9/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 16.5674 - mae: 2.8621 - val_loss: 4.8647 - val_mae: 1.5726
    Epoch 10/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 11.4481 - mae: 2.3008 - val_loss: 2.0312 - val_mae: 0.9791
    Epoch 11/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 11.6064 - mae: 2.3534 - val_loss: 1.8176 - val_mae: 0.9609
    Epoch 12/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 11.9239 - mae: 2.3354 - val_loss: 1.1841 - val_mae: 0.7290
    Epoch 13/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 10.3842 - mae: 2.2587 - val_loss: 0.8873 - val_mae: 0.6126
    Epoch 14/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 8.5337 - mae: 2.0433 - val_loss: 1.9218 - val_mae: 0.9883
    Epoch 15/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 8.2683 - mae: 1.9368 - val_loss: 2.5513 - val_mae: 1.2616
    Epoch 16/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 7.4757 - mae: 1.9414 - val_loss: 4.7696 - val_mae: 1.7738
    Epoch 17/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 7.5790 - mae: 1.9638 - val_loss: 1.7338 - val_mae: 0.9493
    Epoch 18/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 5.4779 - mae: 1.6463 - val_loss: 0.7607 - val_mae: 0.5578
    Epoch 19/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 5.8874 - mae: 1.6093 - val_loss: 0.6785 - val_mae: 0.5752
    Epoch 20/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 5.9742 - mae: 1.5560 - val_loss: 0.5328 - val_mae: 0.4919
    Epoch 21/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.7978 - mae: 1.3412 - val_loss: 0.8203 - val_mae: 0.6037
    Epoch 22/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 3.5253 - mae: 1.2770 - val_loss: 0.6519 - val_mae: 0.5305
    Epoch 23/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 5.4429 - mae: 1.5066 - val_loss: 1.0817 - val_mae: 0.8557
    Epoch 24/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 4.0132 - mae: 1.3731 - val_loss: 0.4549 - val_mae: 0.4790
    Epoch 25/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.9786 - mae: 1.1795 - val_loss: 0.4967 - val_mae: 0.4351
    Epoch 26/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.9524 - mae: 1.1390 - val_loss: 0.5828 - val_mae: 0.4706
    Epoch 27/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 3.5281 - mae: 1.2543 - val_loss: 0.6219 - val_mae: 0.4845
    Epoch 28/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 2.9671 - mae: 1.1463 - val_loss: 0.4259 - val_mae: 0.4048
    Epoch 29/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 3.1091 - mae: 1.1047 - val_loss: 0.4441 - val_mae: 0.5450
    Epoch 30/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.6599 - mae: 1.0244 - val_loss: 0.2997 - val_mae: 0.3390
    Epoch 31/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.3575 - mae: 0.9985 - val_loss: 0.2395 - val_mae: 0.3446
    Epoch 32/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 1.9968 - mae: 0.9792 - val_loss: 0.8765 - val_mae: 0.7231
    Epoch 33/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 3.0959 - mae: 1.1102 - val_loss: 0.3532 - val_mae: 0.3709
    Epoch 34/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 1.9006 - mae: 0.9178 - val_loss: 0.5670 - val_mae: 0.6095
    Epoch 35/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.9484 - mae: 0.9022 - val_loss: 0.5523 - val_mae: 0.6138
    Epoch 36/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2.5523 - mae: 0.9564 - val_loss: 0.4021 - val_mae: 0.5118
    Epoch 37/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.1436 - mae: 0.7529 - val_loss: 0.2377 - val_mae: 0.3749
    Epoch 38/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.2956 - mae: 0.7691 - val_loss: 0.2120 - val_mae: 0.3076
    Epoch 39/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.5725 - mae: 0.8543 - val_loss: 0.4762 - val_mae: 0.5587
    Epoch 40/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.7952 - mae: 0.8063 - val_loss: 0.3692 - val_mae: 0.3840
    Epoch 41/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.9165 - mae: 0.6824 - val_loss: 0.2516 - val_mae: 0.3859
    Epoch 42/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.9667 - mae: 0.6862 - val_loss: 0.2175 - val_mae: 0.3540
    Epoch 43/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.0045 - mae: 0.6419 - val_loss: 0.1733 - val_mae: 0.2895
    Epoch 44/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.8516 - mae: 0.6361 - val_loss: 0.4025 - val_mae: 0.5300
    Epoch 45/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 1.0612 - mae: 0.6297 - val_loss: 0.2774 - val_mae: 0.4221
    Epoch 46/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.6414 - mae: 0.5621 - val_loss: 0.1731 - val_mae: 0.3232
    Epoch 47/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.6542 - mae: 0.5207 - val_loss: 0.1728 - val_mae: 0.3228
    Epoch 48/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.7722 - mae: 0.5610 - val_loss: 0.3048 - val_mae: 0.4035
    Epoch 49/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.9394 - mae: 0.6084 - val_loss: 0.2100 - val_mae: 0.3614
    Epoch 50/50
    [1m21/21[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.6855 - mae: 0.5837 - val_loss: 0.3699 - val_mae: 0.5072
    


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

    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3945 - mae: 0.5268 
    Mean Absolute Error: 0.5125793218612671
    


    
![png](portuguese_files/portuguese_3_1.png)
    



```python
# 예측 저장
predictions = model.predict(X_test)
output = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
output.to_csv('predictions.csv', index=False)

# Kaggle에 제출 (Output 탭에서 다운로드 가능)
```

    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step 
    


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


    
![png](portuguese_files/portuguese_5_0.png)
    



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
    


    
![png](portuguese_files/portuguese_6_1.png)
    



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


    
![png](portuguese_files/portuguese_7_0.png)
    



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


    
![png](portuguese_files/portuguese_8_0.png)
    



```python
# 결과 분포 Boxplot
sns.boxplot(x=y_true, y=y_pred)
plt.title('Predicted Values by True Labels')
plt.xlabel('True Labels')
plt.ylabel('Predicted Probabilities')
plt.show()
```


    
![png](portuguese_files/portuguese_9_0.png)
    



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


    
![png](portuguese_files/portuguese_10_0.png)
    

