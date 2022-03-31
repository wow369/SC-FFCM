import copy
import numpy as np
import pandas as pd
from pandas import *
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import random
import operator
import math
from numpy import *

#每一个Client对象:
class client(object):
    def __init__(self, clusterDataSetFrame, c, init_models, local_max_epoch, cluster_features, server_iter, local_lr, client_control, server_control):
        # 定义
        # m=2
        self.cluster_ds = clusterDataSetFrame

        # 单个client数据集大小
        self.N = len(clusterDataSetFrame)
        # print(self.N)
        self.local_epoch = local_max_epoch
        self.fuzzy_matrix = self.init_fuzzy_matrix(self.N, c)

        self.init_models = np.array(init_models)
        self.models = init_models



        self.cluster_num = c
        self.features = cluster_features
        self.client_gradient = None

        self.server_iter = server_iter
        self.server_control = server_control
        self.lr = local_lr

        self.client_control = client_control

        self.client_control_after = np.zeros(shape=(c, len(cluster_features)))
        # self.client_control = self.client_control_after
        self.delta_c = np.zeros(shape=(c, len(cluster_features)))


        # 单个client运行FCM,更新models，u
        # self.localClustering()

    def init_fuzzy_matrix(self, n_sample, c):
        """
        初始化隶属度矩阵，注意针对一个样本，三个隶属度的相加和=1
        ----
        param n_sample: 样本数量
        param c: 聚类数量
        """
        # 针对数据集中所有样本的隶属度矩阵，shape = [n_sample, c]
        fuzzy_matrix = []
        for i in range(n_sample):
            # 生成 c 个随机数列表, random.random()方法随机生成[0,1)范围内的一个实数。
            random_list = [random.random() for i in range(c)]
            sum_of_random = sum(random_list)
            # 归一化之后的随机数列表
            # 单个样本的模糊隶属度列表
            norm_random_list = [x / sum_of_random for x in random_list]
            # 选择随机参数列表中最大的数的索引
            one_of_random_index = norm_random_list.index(max(norm_random_list))
            for j in range(0, len(norm_random_list)):
                if (j == one_of_random_index):
                    norm_random_list[j] = 1
                else:
                    norm_random_list[j] = 0
            fuzzy_matrix.append(norm_random_list)
        return fuzzy_matrix

    def localClustering(self):
        # print("新一轮：")
        # print(self.models)
        # print("**"*20)
        # print(self.server_control)
        for k in range(0, self.local_epoch):

            self.fuzzy_matrix = update_fuzzy_matrix(self.cluster_ds, self.fuzzy_matrix, self.N, self.cluster_num, 2, self.models)
            self.models = np.array(self.models)
            # print(self.models)
            # print("NN"*50)
            self.get_mini_batch_gradient()
            self.models = self.models - self.lr * (self.client_gradient - self.client_control + self.server_control)


        # 改进：

        self.client_control_after = self.client_control - self.server_control + (1/(self.local_epoch * self.lr)) * (self.init_models - self.models)
        self.delta_c = self.client_control_after - self.client_control
        self.client_control = self.client_control_after


    def get_mini_batch_gradient(self):

        df_values = self.cluster_ds.values
        n_sample = self.N
        c = self.cluster_num
        centers = self.models
        n_features = len(self.features)
        fm = self.fuzzy_matrix
        # nomerator_List = [None] * c
        # print(type(nomerator_List))
        nomerator_List = np.zeros(shape=(c, n_features))

        for s in range(0, c):
            sum_list = np.zeros(n_features)
            for j in range(0, n_features):
                sumJ = 0
                for k in range(0, n_sample):
                    sample = df_values[k]
                    sample = np.array(sample)
                    distance = sample[j] - centers[s][j]
                    res = math.pow(fm[k][s], 2) * distance
                    # sigmoid是一个超参数 此时等于n/10
                    temp = res / 50.5
                    sumJ = sumJ + temp
                sumJ = sumJ * (-2)
                sum_list[j] = sumJ
            nomerator_List[s] = sum_list
        self.client_gradient = nomerator_List


# 更新隶属度矩阵，参考公式 (8)
def update_fuzzy_matrix(df, fuzzy_matrix, n_sample, c, m, cluster_centers):
    # 分母的指数项
    order = float(2 / (m - 1))
    # 遍历样本
    for i in range(n_sample):
        # 单个样本
        sample = list(df.iloc[i])
        # 计算更新公式的分母：样本减去聚类中心
        distances = [np.linalg.norm(np.array(list(map(operator.sub, sample, cluster_centers[j])))) \
                     for j in range(c)]
        for j in range(c):
            # 更新公式的分母
            denominator = sum([math.pow(float(distances[j] / distances[val]), order) for val in range(c)])
            fuzzy_matrix[i][j] = float(1 / denominator)
    return fuzzy_matrix



class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, c, init_models, local_epoch, server_iter, local_lr, client_control,  server_control):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.c = c
        self.initmodels = init_models
        self.max_iter = local_epoch
        self.server_iter = server_iter
        self.lr = local_lr
        # self.dev = dev
        self.clients_set = [None]*numOfClients
        self.client_control = client_control
        self.server_control = server_control
        # self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):


        cluster_data = GetDataSet(self.data_set_name, self.is_iid).cluster_dataFrame
        cluster_features = GetDataSet(self.data_set_name, self.is_iid).features

        # local_data = 505
        # shard_size = 6
        shard_size = len(cluster_data) // self.num_of_clients // 2
        shards_id = np.random.permutation(len(cluster_data) // shard_size)

        for i in range(0, self.num_of_clients):
            # 0 2 4 6...... 偶数
            shards_id1 = shards_id[i * 2]
            # 0+1 = 1 2+1 = 3 .... 奇数
            shards_id2 = shards_id[i * 2 + 1]

            data_shards1 = cluster_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = cluster_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            local_data = np.vstack(((data_shards1, data_shards2)))
            local_data = DataFrame(local_data)

            someone = client(local_data, self.c, self.initmodels, self.max_iter,
                             cluster_features, self.server_iter, self.lr, self.client_control, self.server_control)

            self.clients_set[i] = someone






if __name__=="__main__":
     # MyClient=client()
     # initcenter = [[1.2335556851032348, -0.21070683045092561, -1.2723902502406854, -0.10686097728935547],
     #               [-0.6597260791145286, 0.5042917396898048, 0.5178196833363767, 2.475635174963594],
     #               [1.8084322402343385, -1.188538845764442, -1.0990303014313072, 0.15537622465817208]]
     # MyClients = ClientsGroup('housing', True, 40, 0.1, 3, initcenter, 5, 0)
     # MyClient = MyClients.clients_set[0]

     None



