import copy
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import random
from clients import ClientsGroup, client
from getData import GetDataSet


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients客户机数量')
parser.add_argument('-cf', '--cfraction', type=float, default=0.5, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=50, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=126, help='local train batch size本地每次放入数据量')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.5, help="learning rate, \
                    use value from origin paper as default")

parser.add_argument('-sr', "--Server_learning_rate", type=float, default=0.1, help="Server learning rate")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)模型验证频率（通信）")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)全局模型保存频率（通信）')
parser.add_argument('-ncomm', '--num_comm', type=int, default=500, help='number of communications通讯次数')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
parser.add_argument('-Dn', '--cluster_dataSet_name', type=str, default='housing', help='聚类数据名称')
parser.add_argument('-Cn', '--number_of_cluster', type=int, default=3, help='聚类数量')
parser.add_argument('-Dnum', '--number_of_data', type=int, default=505, help='数据数量')
parser.add_argument('-nf', '--number_of_features', type=int, default=3, help='特征数量')


# def test_mkdir(path):
#     if not os.path.isdir(path):
#         os.mkdir(path)

#初始化原型
def init_FFCM_models(n_features, c, init_method='random'):
    init_cluster_centers = []
    #cluster_centers = []
    n_features = n_features
    if init_method == 'multi_normal':
        # 均值列表
        mean = [0] * n_features
        # 多元高斯分布的协方差矩阵，对角阵 求得原型
        cov = np.identity(n_features)
        for i in range(0, c):
            init_cluster_centers.append(list(np.random.multivariate_normal(mean, cov)))
        #     else:
        #         init_cluster_centers = [[0.1] * n_features ] * c
    return init_cluster_centers

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    models = None
    f_num = args['number_of_features']
    P = int(copy.deepcopy(args['num_of_clients']) * args["cfraction"])
    data_set_name = args['cluster_dataSet_name']
    local_epoch = args['epoch']



    # Alpha = args["Server_learning_rate"]

    current_iter = 0
    c = args['number_of_cluster']
    initcenter = init_FFCM_models(f_num, args['number_of_cluster'], init_method='multi_normal')
    # initcenter = np.ones(shape=(c, f_num))

    server_control = np.zeros(shape=(c, f_num))
    client_control = np.zeros(shape=(c, f_num))

    # 交流开始
    cumm = args["num_comm"]

    # 初始化clients集
    models = initcenter


    myClientsGroup1 = ClientsGroup(args["cluster_dataSet_name"], args['IID'], args['num_of_clients'],
                             args['number_of_cluster'], models,
                             local_epoch, current_iter, args["local_learning_rate"], client_control, server_control)

    # clientSet1 = myClientsGroup1.clients_set[:P]
    # clientSet2 = myClientsGroup1.clients_set[P:]

    # seq = [clientSet1, clientSet2]


    for current_iter in range(0, cumm):

        # 以下计算需要在ndArray下执行
        # models = np.array(models)
        i = random.randint(0, 1)
        # print(i)
        myClients = myClientsGroup1.clients_set

        # Server端平均模型参数V（models）
        print("第", current_iter, "轮开始：")
        print("--" * 50)
        temp = np.zeros(shape=(c, f_num))

        Beta = 1 / P
        temp_c = np.zeros(shape=(c, f_num))

        for i in range(0, P):
            delta_v = 0
            client = random.randint(0, 9)
            myClient = myClients[client]
            models = np.array(models)

            myClient.models = models
            myClient.init_models = models
            myClient.server_control = server_control

            myClient.localClustering()
            client_models = np.array(myClient.models)
            # print(myClient.cluster_ds[:5])
            # print(client_models)

            # print(models)
            delta_v = client_models - models
            # print("差：")
            # print(delta_v)

            temp_c = temp_c + myClient.delta_c
            client_control = myClient.client_control
            # temp_c = client_control

            temp = temp + delta_v
            print("--" * 50)
            print(np.array(myClient.models))
            print("--" * 50)

        delta_c = temp_c * Beta
        server_control = server_control + args["cfraction"] * delta_c
        # print(server_control)

        delta_data = 0
        server_step = args["Server_learning_rate"]
        delta_data = temp * Beta
        models = models + (delta_data * server_step)
        # print("第",current_iter,"轮结果：","**"*50)
        # print(models)

        print("第", current_iter, "轮结果：", "**" * 50)
        endModels = models
        endModels = endModels[np.argsort(endModels[:, -1]), :]
        print(endModels)
    print("最终结果：")
    print(models)
    # None








