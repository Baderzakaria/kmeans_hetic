import numpy as np
import matplotlib.pyplot as plt

def k_means(data, n_clusters, max_iter=100):
    centers = np.random.rand(n_clusters, data.shape[1])
    outputs = np.zeros(data.shape[0])
    total_centers = []

    for iter in range(max_iter):
        prev_centers = centers.copy()

        for i in range(data.shape[0]):
            distances = np.linalg.norm(data[i, :] - centers, axis=1)
            outputs[i] = np.argmin(distances)
        for c in range(n_clusters):
            cluster_data = data[outputs == c]
            centers[c] = np.mean(cluster_data, axis=0) if len(cluster_data) > 0 else prev_centers[c]
            total_centers.append(centers[c])

        if np.linalg.norm(centers - prev_centers) < 0.01:
            break
    return centers, outputs, total_centers

def draw_clusters(data, centers, outputs, total_centers):
    cmap = plt.cm.get_cmap('tab10')

    plt.figure(figsize=(8, 6))

    for i in range(centers.shape[0]):
        cluster_data = data[outputs == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cmap(i), label=f'Cluster {i + 1}') 
    for i in range(len(total_centers)):
        plt.scatter(total_centers[i][0], total_centers[i][1], c='black', marker='o')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Final Centers')
    plt.legend()
    plt.show()

def calculate_wcss(data, centers, outputs):
    wcss = 0
    for i in range(centers.shape[0]):
        cluster_data = data[outputs == i]
        if len(cluster_data) > 0:
            wcss += np.sum(np.square(cluster_data - centers[i]))
    return wcss

class_1 = np.random.rand(200, 2) * 50
class_2 = np.random.rand(200, 2) * 50 + 4
class_3 = np.random.rand(200, 2) * 60 - 2

data = np.vstack([class_1, class_2, class_3])
np.random.shuffle(data)

max_k = 10
wcss_values = []
for k in range(1, max_k + 1):
    centers, outputs, total_centers = k_means(data, k)
    draw_clusters(data, centers, outputs, total_centers)
    wcss = calculate_wcss(data, centers, outputs)
    wcss_values.append(wcss)

plt.plot(range(1, max_k + 1), wcss_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.show()
