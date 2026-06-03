import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. 와인 데이터셋 읽기
wine = datasets.load_wine()

# pandas 데이터프레임(표)으로 변환
# data에는 13가지 화학 성분 수치를, columns에는 성분 이름들을 넣음
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# 표의 맨 오른쪽 끝에 정답지(Target, 품종 클래스 번호) 추가
wine_df['target'] = wine.target

# 상위 5개 행만 표로 출력
print(wine_df.head())


# 2. 데이터셋 분리
X = wine.data
y = wine.target

# (80:20)으로 분할한다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# 3. 모델 선택 및 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 스케일러를 불러와서 훈련 데이터에 맞게 학습 후 변환
X_test_scaled = scaler.transform(X_test) # 테스트 데이터도 똑같은 기준으로 변환

# 스케일링된 데이터로 모델을 학습
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_scaled, y_train)



# 4. 모델 평가
y_pred = knn.predict(X_test_scaled) # 스케일링된 테스트 데이터로 평가
scores = metrics.accuracy_score(y_test, y_pred)

print("scores:", scores)


# 5. 결과 예측
# 와인 데이터의 타겟 이름 정의
classes = {0: 'class_0', 1: 'class_1', 2: 'class_2'}

# 새로운 데이터 제시
x_new = [
    [14.5, 1.8, 2.5, 15.5, 115.0, 2.9, 3.2, 0.3, 2.4, 6.5, 1.0, 3.8, 1150.0],  # class_0 추정
    [12.5, 1.2, 1.4, 11.5,  90.0, 2.0, 0.8, 0.3, 0.6, 2.2, 1.1, 2.0,  450.0],  # class_1 추정
    [13.5, 3.6, 2.4, 20.0, 105.0, 1.6, 0.9, 0.4, 1.1, 8.5, 0.6, 1.3,  650.0]   # class_2 추정
]

# 모델이 스케일링된 데이터로 학습되었으므로, 새로운 데이터(x_new)도 반드시 동일한 기준으로 스케일링(변환)을 거쳐야 한다
# (fit_transform이 아닌 scaler.transform()을 사용)
x_new_scaled = scaler.transform(x_new)

# 스케일링 완료된 데이터로 예측 수행
y_predict = knn.predict(x_new_scaled)

print("첫 번째 와인 예측:", classes[y_predict[0]])
print("두 번째 와인 예측:", classes[y_predict[1]])
print("세 번째 와인 예측:", classes[y_predict[2]])