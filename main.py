import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)


# 4. 모델 평가
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print("scores:", scores)