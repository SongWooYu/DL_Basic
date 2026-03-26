import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import joblib
from collections import Counter

# 원본 데이터
dach_length = [77, 78, 85, 83, 73, 77, 73, 80]
dach_height = [25, 28, 29, 30, 21, 22, 17, 35]

samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

d_data = np.column_stack((dach_length, dach_height))
s_data = np.column_stack((samo_length, samo_height))

X = np.concatenate((d_data, s_data))
y_true = np.array(['Dachshund'] * len(d_data) + ['Samoyed'] * len(s_data))

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
# 모델 저장
model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'kmeans_model.pkl')
joblib.dump(kmeans, model_path)

cluster_labels = kmeans.labels_


# 클러스터 중심의 y값(키) 기준으로 항상 0: Dachshund, 1: Samoyed로 매핑
centers = kmeans.cluster_centers_
if centers[0, 1] < centers[1, 1]:
    cluster_to_dog = {0: 'Dachshund', 1: 'Samoyed'}
else:
    cluster_to_dog = {0: 'Samoyed', 1: 'Dachshund'}


def predict_dog(length, height):
    new_point = np.array([[length, height]])
    cluster = kmeans.predict(new_point)[0]
    dog_name = cluster_to_dog[cluster]
    return dog_name, cluster


def draw_graph(length, height, save_path='plots/dog_plot.png'):
    plt.figure(figsize=(8, 6))

    # 원본 데이터
    plt.scatter(d_data[:, 0], d_data[:, 1], label='Dachshund', s=80)
    plt.scatter(s_data[:, 0], s_data[:, 1], label='Samoyed', s=80)

    # 중심점
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=250, label='Centers')

    # 새 입력 데이터
    dog_name, cluster = predict_dog(length, height)
    plt.scatter(length, height, marker='*', s=300, label=f'New Dog: {dog_name}')

    plt.xlabel('Length')
    plt.ylabel('Height')
    plt.title('Dog Clustering with KMeans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return dog_name