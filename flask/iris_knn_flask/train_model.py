import pickle
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

import os
os.makedirs('model', exist_ok=True)
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("모델이 model/iris_model.pkl에 저장되었습니다.")