import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import os
import joblib

# 모델 경로
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
model_path = os.path.join(model_dir, 'kmeans_model.pkl')

# 데이터 (모델이 없을 때만 사용)
dach_length = [77, 78, 85, 83, 73, 77, 73, 80]
dach_height = [25, 28, 29, 30, 21, 22, 17, 35]
samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]
d_data = np.column_stack((dach_length, dach_height))
s_data = np.column_stack((samo_length, samo_height))
X = np.concatenate((d_data, s_data))
y_true = np.array(['Dachshund'] * len(d_data) + ['Samoyed'] * len(s_data))

# 모델 로드 또는 학습
if os.path.exists(model_path):
    kmeans = joblib.load(model_path)
else:
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, model_path)

cluster_labels = kmeans.labels_ if hasattr(kmeans, 'labels_') else kmeans.predict(X)
cluster_to_dog = {}
for cluster_id in range(2):
    dog_names = y_true[cluster_labels == cluster_id]
    most_common = Counter(dog_names).most_common(1)[0][0]
    cluster_to_dog[cluster_id] = most_common

def predict_dog(length, height):
    new_point = np.array([[length, height]])
    cluster = kmeans.predict(new_point)[0]
    dog_name = cluster_to_dog[cluster]
    return dog_name, cluster

def draw_graph(length, height, save_path='plots/dog_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.scatter(d_data[:, 0], d_data[:, 1], label='Dachshund', s=80)
    plt.scatter(s_data[:, 0], s_data[:, 1], label='Samoyed', s=80)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=250, label='Centers')
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
