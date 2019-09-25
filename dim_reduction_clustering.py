from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D


path = './new_data2/data_feature3.csv'


def load_data(path):
    f = open(path, "rb")
    data = np.loadtxt(f, delimiter=',', skiprows=0)
    f.close()
    data1 = np.array(data)
    return data1


data = load_data(path)
# print(load_data(path))
data1 = np.delete(data, 0, axis=1)  # 去掉数据第一列，短行程时长
# t-SNE 方法降维，降成3维，固定随机种子random_state
# t-distributed Stochastic Neighbor Embedding(t-SNE)
model = TSNE(n_components=3, learning_rate=100, random_state=30)

# 降维后的矩阵 3列
transformed = model.fit_transform(data1)
# print(transformed)


# k-means聚类
def k_means_clustering(data):
    estimator = KMeans(n_clusters=3, random_state=1)  # 构造聚类器
    estimator.fit(data)  # 聚类
    #print(data)
    label_pred = estimator.labels_  # 获取聚类标签
    cluster_centers = estimator.cluster_centers_
    # 绘制k-means结果
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]
    # print(x0)
    x00 = x0[:, 0]
    y00 = x0[:, 1]
    z00 = x0[:, 2]
    x11 = x1[:, 0]
    y11 = x1[:, 1]
    z11 = x1[:, 2]
    x22 = x2[:, 0]
    y22 = x2[:, 1]
    z22 = x2[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x00, y00, z00, c="red", marker='o', label='label0')
    ax.scatter(x11, y11, z11, c="green", marker='*', label='label1')
    ax.scatter(x22, y22, z22, c="blue", marker='+', label='label2')
    '''plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')'''
    plt.legend(loc=2)
    plt.show()
    return x0, x1, x2, cluster_centers


# 层次聚类
def hierarchical_clustering(data):
    # 设置分层聚类函数
    linkages = ['ward', 'average', 'complete']
    n_clusters_ = 3
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
    # 训练数据
    ac.fit(data)
    # 每个数据的分类
    label_pred = ac.labels_
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]

    x00 = x0[:, 0]
    y00 = x0[:, 1]
    z00 = x0[:, 2]
    x11 = x1[:, 0]
    y11 = x1[:, 1]
    z11 = x1[:, 2]
    x22 = x2[:, 0]
    y22 = x2[:, 1]
    z22 = x2[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x00, y00, z00, c="red", marker='o', label='label0')
    ax.scatter(x11, y11, z11, c="green", marker='*', label='label1')
    ax.scatter(x22, y22, z22, c="blue", marker='+', label='label2')
    '''plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')'''
    plt.legend(loc=2)
    plt.show()
    return x0, x1, x2


# Birch(利用层次方法的平衡迭代规约和聚类)
def birch_clustering(data):
    # 设置birch函数
    birch = Birch(n_clusters=3)
    # 训练数据
    birch.fit_predict(data)
    label_pred = birch.labels_
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]

    x00 = x0[:, 0]
    y00 = x0[:, 1]
    z00 = x0[:, 2]
    x11 = x1[:, 0]
    y11 = x1[:, 1]
    z11 = x1[:, 2]
    x22 = x2[:, 0]
    y22 = x2[:, 1]
    z22 = x2[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x00, y00, z00, c="red", marker='o', label='label0')
    ax.scatter(x11, y11, z11, c="green", marker='*', label='label1')
    ax.scatter(x22, y22, z22, c="blue", marker='+', label='label2')
    '''plt.scatter(x0[:, 1], x0[:, 2], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 1], x1[:, 2], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 1], x2[:, 2], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')'''
    plt.legend(loc=2)
    plt.show()
    return x0, x1, x2


class0, class1, class2, cluster_centers = k_means_clustering(transformed)
# hierarchical_clustering(transformed)
# birch_clustering(transformed)

total_data = transformed
count0 = class0.shape[0]
count1 = class1.shape[0]
count2 = class2.shape[0]

n = total_data.shape[0]


def extraction(count, sub_class, fa_class, original_data):
    storke_class = [0] * 14
    storke_class_np = np.array(storke_class)
    for i in range(count):
        for j in range(n):
            result = ((sub_class[i]==fa_class[j]).all())
            if result:
                storke_class_np = np.vstack((storke_class_np, original_data[j]))

    np.delete(storke_class_np, 0, axis=0)
    return storke_class_np


storke_class0 = extraction(count0, class0, total_data, data)
storke_class1 = extraction(count1, class1, total_data, data)
storke_class2 = extraction(count2, class2, total_data, data)
np.savetxt('./new_data2/3_tSNE_class0.csv', class0, delimiter=',', fmt='%.4f')
np.savetxt('./new_data2/3_tSNE_class1.csv', class1, delimiter=',', fmt='%.4f')
np.savetxt('./new_data2/3_tSNE_class2.csv', class2, delimiter=',', fmt='%.4f')
np.savetxt('./new_data2/3_storke_class0.csv', storke_class0, delimiter=',', fmt='%.4f')
np.savetxt('./new_data2/3_storke_class1.csv', storke_class1, delimiter=',', fmt='%.4f')
np.savetxt('./new_data2/3_storke_class2.csv', storke_class2, delimiter=',', fmt='%.4f')

print(storke_class0.shape[0])
print(storke_class1.shape[0])
print(storke_class2.shape[0])
# print(storke_class0)
# print(storke_class1)
# print(storke_class2)
print(cluster_centers)

