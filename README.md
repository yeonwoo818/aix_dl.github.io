# AIX_DL.github.io


[MEMBERS]

성연우 (생명과학과)
임종호 (기계공학부) 

1. PROPOSAL

   [Motivation: why are you doing this?]

   [What do you want to see at the end?]

2. DATASETS

   [Alcohol effects on study]
   - 해당 데이터는 두 포르투갈 학교의 중등교육 학생 성취도를 다룬다. Math course와 Portuguese language course를 수강하는 학생들의 기본 정보(성별, 거주지, 부모님 직업 등) 및 알코올 섭취 정도, 성적을 포함한다.
   - 두 과목 모두 학생 기본 정보 및 알코올 섭취 정도를 나타내는 30개의 열과 성적을 나타내는 3개의 열로 구성되어있으며, Math course의 경우 395개, Portuguese language course의 경우 649개의 데이터 셋으로 구성되어있다.
   - 데이터 셋은 이진 분류, 5단계 분류, 회귀 방법에 따라 모델링 되었다.

      - 이진분류(binary classification): 입력값에 따라 분류한 카테고리가 두 가지인 분류 알고리즘. 
      - 다중분류(5-level classification): 입력값에 따라 분류한 카테고리가 세 가지 이상(본 데이터셋에서는 다섯 가지)인 분류 알고리즘.
      - 회귀(regression):

  |Columns|설명|classification level|
  |---|---|----|
  |school|학생의 학교. 'GP' - Gabriel Pereira / 'MS' - Mousinho da Silveira|binary|
  |sex|학생의 성별. 'F' - 여성 / 'M' - 남성|binary|
  |age|학생의 나이. 15세 - 22세|numeric|
  |adress|학생의 거주지. 'U' - 도시 / 'R' - 지방|binary|
  |famsize|학생의 가족 구성원 수. 'LE3' - 3명 이하 / 'GT3' - 4명 이상|binary|
  |Pstatus|부모 동거 상태. 'T' - 동거 / 'A' - 별거|binary|
  |Medu|모 교육 정도. 0 - 없음 / 1 - 초등교육(~4학년) / 2 - 초등교육(5~9학년) / 3 - 중등교육 / 4 - 고등교육|numeric|
  |Fedu|부 교육 정도. 0 - 없음 / 1 - 초등교육(~4학년) / 2 - 초등교육(5~9학년) / 3 - 중등교육 / 4 - 고등교육|numeric|
  |Mjob|모 직업. '교사', '보건', '공무원', '없음', '기타'|nominal|
  |Fjob|부 직업. '교사', '보건', '공무원', '없음', '기타'|nominal|
  |reason|학교를 선택한 이유. ~~~|nominal|
  |guardian|학생의 보호자. 'mother', 'father', 'other'|nominal|
  |traveltime|통학시간. 1 - 15분 이하 / 2 - 15~30분 / 3 - 30~60분 / 4 - 60분 이상|numeric|
  |studytime|주간 학습 시간. 1 - 2시간 이하 / 2 - 2~5시간 / 3 - 5~10시간 / 4 - 10시간 이상|numeric|
  |failures|~~~~|numeric|
  |schoolsup|~~~|binary|

>> 근데 이게 맞냐 가독성 너무 안좋은데 항목도 14개 더 써야함
  

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




4. EVALUATION & ANALYSIS
5. RELATED WORK
6. CONCLUSION: DISCUSSION
