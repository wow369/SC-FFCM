import copy
import numpy as np
import pandas as pd
from pandas import *
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from skfuzzy.cluster import cmeans
import random
import operator
import math
from numpy import *

#每一个Client对象:
class client(object):
    def __init__(self, clusterDataSetFrame, c, init_models, local_max_epoch, cluster_features, server_iter, local_lr, client_control, server_control, m):
        # 定义
        self.cluster_ds = clusterDataSetFrame

        # 单个client数据集大小
        self.N = len(clusterDataSetFrame)
        self.m = m
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

        self.support_centers = init_models


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

        # 每个客户端使用自己的本地数据运行一次完整的模糊C均值聚类算法，生成本地的聚类中心
        def preInit(self):
            # self.cluster_ds 是客户端的本地数据，形状通常是 [样本数, 特征数]，cmeans 函数要求输入数据格式为 [特征数, 样本数]
            df_T = self.cluster_ds.T
            # cmeans 函数返回值：cntr, u, u0, d, jm, p, fpc = cmeans(...)
            '''
            cntr: 聚类中心，形状为 (c, n_features)
            u: 隶属度矩阵，形状为 (c, n_samples)
            u0: 初始隶属度矩阵
            d: 样本到聚类中心的距离矩阵
            jm: 目标函数值的历史记录
            p: 迭代次数
            fpc: 模糊分区系数，衡量聚类效果的指标
            '''
            self.support_centers, _, _, _, _, _, _ = cmeans(df_T, c=self.cluster_num, m=self.m, error=0.0005,
                                                            maxiter=100)

    def localClustering(self):
        # print("新一轮：")
        # print(self.models)
        # print("**"*20)
        # print(self.server_control)
        for k in range(0, self.local_epoch):

            self.fuzzy_matrix = update_fuzzy_matrix(self.cluster_ds, self.fuzzy_matrix, self.N, self.cluster_num, self.m, self.models)
            # models 为中心点, 聚类数量和客户端数量都为 10 (合成数据集下)
            self.models = np.array(self.models)
            self.get_mini_batch_gradient()
            # 公式(14)
            self.models = self.models - self.lr * (self.client_gradient - self.client_control + self.server_control)

        # 改进：(公式16, 18)
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

        # 初始化 nomerator_List
        nomerator_List = np.zeros((c, n_features))

        # 预先计算 fm^m，避免每次都计算, m 为模糊指数
        fm_pow_m = np.power(fm, self.m)

        # 对每个聚类中心进行操作 (公式13)
        for k in range(c):
            # 对每个特征维度进行计算
            for j in range(n_features):
                # 计算梯度的分子
                distance = df_values[:, j] - centers[k][j]  # 向量化计算距离
                res = fm_pow_m[:, k] * distance  # 通过向量化计算距离的加权结果

                # 计算 sumJ
                sumJ = np.sum(res) / n_sample
                nomerator_List[k, j] = -2 * sumJ  # 更新 nomerator_List 中的值

        # 将计算结果赋值给 client_gradient
        self.client_gradient = nomerator_List
    # 公式(9) 计算聚类目标函数值
    def get_obj(self, c, res_centers):

        m = self.m
        n_sample = self.N
        df_values = self.cluster_ds.values
        vali_fuzzyMatrix = update_fuzzy_matrix(self.cluster_ds, self.fuzzy_matrix, self.N, self.cluster_num, m,
                                               res_centers)
        temp_c_obj = 0

        for k in range(0, c):
            res_k = res_centers[k]
            # print(res_k)
            temp_n_obj = 0
            for s in range(0, n_sample):
                sample = np.array(df_values[s])
                # print('sample', sample)
                dis = sum(np.power((sample - res_k), 2))
                # print('dis', dis)
                obj_s = np.power(vali_fuzzyMatrix[s][k], m) * dis
                temp_n_obj = temp_n_obj + obj_s
            # print('temp_n_obj',temp_n_obj)
            temp_c_obj = temp_c_obj + temp_n_obj

        return temp_c_obj


# 更新隶属度矩阵，参考公式 (8)  公式 (12) 吧
def update_fuzzy_matrix(df, fuzzy_matrix, n_sample, c, m, cluster_centers):
    # 分母的指数项
    order = float(2 / (m - 1))
    # 遍历样本
    for i in range(n_sample):
        # 单个样本, .iloc[i] 表示取第 i 行的数据（i 从 0 开始）, 返回的是该行的所有特征值
        sample = list(df.iloc[i])
        # 计算更新公式的分母：样本减去聚类中心
        # 将当前样本的每个特征值与第 j 个聚类中心的对应特征值做相减，得到一个“差值向量”
        # operator.sub：是 Python 标准库 operator 模块中的减法函数，相当于 lambda a, b: a - b。
        distances = [np.linalg.norm(np.array(list(map(operator.sub, sample, cluster_centers[j])))) \
                     for j in range(c)]
        for j in range(c):
            # 更新公式的分母
            denominator = sum([math.pow(float(distances[j] / distances[val]), order) for val in range(c)])
            fuzzy_matrix[i][j] = float(1 / denominator)
    return fuzzy_matrix

class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, c, init_models, local_epoch, server_iter, local_lr, client_control,  server_control,m):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.c = c
        self.initmodels = init_models
        self.max_iter = local_epoch
        self.server_iter = server_iter
        self.lr = local_lr
        self.clients_set = [None]*numOfClients
        self.client_control = client_control
        self.server_control = server_control
        self.m = m

        if self.data_set_name == '2D':
            if self.is_iid == 1:
                self.syn_iid_data_Allocation()

            else:
                self.syn_non_iid_data_Allocation()
        else:
            if self.is_iid == 1:
                self.dataSetAllocation()

            else:
                self.dataSetBalanceAllocation()

    def dataSetAllocation(self):

        cluster_data = GetDataSet(self.data_set_name, self.is_iid).cluster_dataFrame
        cluster_features = GetDataSet(self.data_set_name, self.is_iid).features

        shard_size = len(cluster_data) // self.num_of_clients
        shards_id = np.random.permutation(len(cluster_data) // shard_size)

        for i in range(0, self.num_of_clients):
            shards_idx = shards_id[i]
            # 0+1 = 1 2+1 = 3 .... 奇数
            # shards_id2 = shards_id[i * 2 + 1]

            data_shards = cluster_data[shards_idx * shard_size: shards_idx * shard_size + shard_size]
            # data_shards2 = cluster_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            if shards_idx == 9:
                data_shards = cluster_data[shards_idx * shard_size: shards_idx * shard_size + shard_size +
                                                                    (len(cluster_data) - (
                                                                            shards_idx * shard_size + shard_size))]

            local_data = data_shards
            local_data = DataFrame(local_data)

            someone = client(local_data, self.c, self.initmodels, self.max_iter,
                             cluster_features, self.server_iter, self.lr, self.client_control,
                             self.server_control,self.m)

            self.clients_set[i] = someone

    # 数据分片分配，将完整的数据集平衡地分配给多个客户端，实现 Non-IID 的数据分布。
    def dataSetBalanceAllocation(self):

        cluster_data = GetDataSet(self.data_set_name, self.is_iid).cluster_dataFrame
        cluster_features = GetDataSet(self.data_set_name, self.is_iid).features

        # local_data = 150
        # shard_size = 6
        # shard_size = len(cluster_data) // self.num_of_clients // 2
        shard_size = len(cluster_data) // self.num_of_clients
        # print('shard_size',shard_size)
        shards_ids = np.random.permutation(len(cluster_data) // shard_size)
        # shards_ids = np.arange(0, self.num_of_clients)
        print('shards id', shards_ids)

        for i in range(0, self.num_of_clients):
            # 0 2 4 6...... 偶数
            # shards_id1 = shards_id[i * 2]
            # 0+1 = 1 2+1 = 3 .... 奇数
            # shards_id2 = shards_id[i * 2 + 1]
            shards_idx = shards_ids[i]

            # data_shards1 = cluster_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # data_shards2 = cluster_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            data_shards = cluster_data[shards_idx * shard_size: shards_idx * shard_size + shard_size]

            if shards_idx == 9:
                data_shards = cluster_data[shards_idx * shard_size: shards_idx * shard_size + shard_size +
                                                                    (len(cluster_data) - (
                                                                            shards_idx * shard_size + shard_size))]

            # local_data = np.vstack(((data_shards1, data_shards2)))
            local_data = data_shards
            local_data = DataFrame(local_data)

            someone = client(local_data, self.c, self.initmodels, self.max_iter,
                             cluster_features, self.server_iter, self.lr, self.client_control,
                             self.server_control, self.m)

            self.clients_set[i] = someone

    def syn_iid_data_Allocation(self):

        self.num_of_clients = 5

        for i in range(0, self.num_of_clients):

            local_data = np.load(f'./data/clients_data/client_{i}_iid_data.npy')
            local_data = DataFrame(local_data)

            columns = list(local_data.columns)
            features = columns[:len(columns) - 1]
            ex_df = local_data[features]


            someone = client(ex_df, self.c, self.initmodels, self.max_iter,
                             features, self.server_iter, self.lr, self.client_control, self.server_control,self.m)

            self.clients_set[i] = someone


    # Non-IID
    def syn_non_iid_data_Allocation(self):

        self.num_of_clients = 5

        for i in range(0, self.num_of_clients):

            local_data = np.load(f'./data/clients_data/client_{i}_non-iid_data.npy')
            local_data = DataFrame(local_data)

            columns = list(local_data.columns)
            features = columns[:len(columns) - 1]
            ex_df = local_data[features]

            someone = client(ex_df, self.c, self.initmodels, self.max_iter,
                             features, self.server_iter, self.lr, self.client_control, self.server_control,self.m)

            self.clients_set[i] = someone


if __name__=="__main__":


     None



