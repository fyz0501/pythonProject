from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import random

class Cluster:

    # 求出样本均值
    def cal_centre(self, nodes, dim):
        ave = np.zeros(dim, float)
        for n in nodes:
            for i in range(0, dim):
                ave[i] += n[i]
        for i in range(0, dim):
            ave[i] = ave[i] / len(nodes)
        return ave

    # 根据距离进行划分
    def classify(self, centre, x):
        dis = []
        for node in centre:
            dis.append(distance.euclidean(x, node))  # 采用欧氏距离
        min_dis, min_index = np.min(dis), np.argmin(dis)
        return min_dis, min_index

    # 顺序聚类
    def Sequential(self, X, theta, classnum):
        dim = len(X[0])    # 样本的特征维度
        cluster_list = [[] for i in range(0, classnum)]
        cluster_list[0].append(X[0])
        class_cnt = 1
        for i in range(1, len(X)):
            centre = [self.cal_centre(cluster_list[j], dim) for j in range(0, class_cnt)] # 聚类中心
            dis, class_index = self.classify(centre, X[i])
            if class_cnt < classnum and dis > theta:
                cluster_list[class_cnt].append(X[i])
                class_cnt += 1
            else:
                cluster_list[class_index].append(X[i])
        return cluster_list

    # 计算聚类X的Dunn指数
    def Dunn(self, X):
        diams = []
        for i in range(0, len(X)):
            cluster_dim = []
            for j in range(0, len(X[i])):
                for k in range(j+1, len(X[i])):
                    cluster_dim.append(distance.euclidean(X[i][j], X[i][k]))
            max_dim = max(cluster_dim)
            diams.append(max_dim)
        diam = max(diams)
        cluster_dis = 10000000.0
        for i in range(0, len(X)):
            for j in range(i+1, len(X)):
                for k in range(0, len(X[i])):
                    for l in range(0, len(X[j])):
                        dis = distance.euclidean(X[i][k], X[j][l])
                        if cluster_dis > dis:
                            cluster_dis = dis
        print("Dunn指数为：{}\n".format(cluster_dis/diam))
        return cluster_dis/diam

    # 计算聚类X的Davies-Bouldin指数
    def Davies_Bouldin(self, X):
        dim = len(X[0][0])
        R = [[] for i in range(0, len(X))]
        for i in range(0, len(X)):
            for j in range(i+1, len(X)):
                m_i = self.cal_centre(X[i], dim)
                m_j = self.cal_centre(X[j], dim)
                d_ij = distance.euclidean(m_i, m_j)
                s_i = np.sqrt(sum([distance.euclidean(X[i][k], m_i)**2 for k in range(0, len(X[i]))]) / len(X[i]))
                s_j = np.sqrt(sum([distance.euclidean(X[j][k], m_j)**2 for k in range(0, len(X[j]))]) / len(X[j]))
                R[i].append((s_i + s_j)/d_ij)
                R[j].append((s_i + s_j)/d_ij)
        J = 0
        for i in range(0, len(X)):
            J += max(R[i])
        print("DB指数为：{}\n".format(J/len(X)))
        return J/len(X)

    # 谱系聚类
    def Spectral(self, X, classnum):
        dim = len(X[0])
        class_cnt = len(X)
        cluster_list = [[] for i in range(0, len(X))]
        for i in range (0, len(X)):
            cluster_list[i].append(X[i])
        while class_cnt > classnum:
            centre = [self.cal_centre(cluster_list[i], dim) for i in range(0, class_cnt)]  # 聚类中心
            min_dis = 1000000.0
            min_index_1 = -1
            min_index_2 = -1
            for i in range (0, class_cnt):
                for j in range(i+1, class_cnt):
                    dis = distance.euclidean(centre[i], centre[j])
                    if dis < min_dis:
                        min_dis = dis
                        min_index_1, min_index_2 = i, j
            cluster_list[min_index_1].extend(cluster_list[min_index_2])
            cluster_list.pop(min_index_2)
            class_cnt -= 1
        return cluster_list

    # K-means
    # init_centre为外部传进来的初始聚类中心，可以通过其获取到聚类的数量
    def K_means(self, X, init_centre):
        centre = init_centre
        dim = len(X[0])
        flag = True
        cluster_list = [[] for i in range(0, len(init_centre))]
        while flag:
            cluster_list = [[] for i in range(0, len(init_centre))]
            last_centre = centre
            for i in range(0, len(X)):
                dis, class_index = self.classify(centre, X[i])  # 分类
                cluster_list[class_index].append(X[i])
            centre = [self.cal_centre(cluster_list[i], dim) for i in range(0, len(init_centre))]
            sub = (np.array(last_centre) - np.array(centre))
            flag = sub.all()
        return cluster_list

    def gen_data(self, mean, cov, num):
        data = np.random.multivariate_normal(mean, cov, num)
        return data

    def best_cluster_num(self, X):
        dim = len(X[0])
        dis_list = []
        for hyp_k in range(1, 10):
            # centre = np.random.choice(8, (hyp_k, 2), replace=False).tolist()  # 这里其实没封装好
            centre = []
            for i in range(0, hyp_k):
                index = random.randint(0, 200)
                centre.append(X[(index+i*200)%600])
            cluster_list = self.K_means(X, centre)
            dis = 0
            for i in range(0, len(cluster_list)):
                m_i = self.cal_centre(cluster_list[i], dim)
                s_i = sum([distance.euclidean(cluster_list[i][k], m_i) ** 2 for k in range(0, len(cluster_list[i]))]) / len(cluster_list[i])
                dis += s_i
            dis_list.append(dis/hyp_k)
        #return dis_list
        plt.title('Intraclass distance')
        plt.plot(range(1, 10), dis_list, '*-', label="Intraclass distance")
        plt.grid(True)
        plt.legend()
