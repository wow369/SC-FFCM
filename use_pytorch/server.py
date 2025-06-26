import copy
import math
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
import operator
from clients import ClientsGroup, client
from getData import GetDataSet
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import time
from skfuzzy.cluster import cmeans

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.5,
                    help='C fraction, 0 means 1 client, 1 means total clients')

parser.add_argument('-E', '--epoch', type=int, default=50, help='local train epoch')

parser.add_argument('-sr', "--Server_learning_rate", type=float, default=0.5, help="Server learning rate")
parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.2, help="local learning rate")

parser.add_argument('-ncomm', '--num_comm', type=int, default=500, help='number of communications')
# parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

parser.add_argument('-init_meth', '--init_method', type=str, default='co_init', help='init method')
parser.add_argument('-fi', '--fuzzy_index', type=float, default=2, help='fuzzy index')

# parser.add_argument('-Dn', '--cluster_dataSet_name', type=str, default='iris', help='dataset name')
# parser.add_argument('-Cn', '--number_of_cluster', type=int, default=3, help='cluster number')
# parser.add_argument('-Dnum', '--number_of_data', type=int, default=150, help='sample number')
# parser.add_argument('-nf', '--number_of_features', type=int, default=4, help='feature number')


parser.add_argument('-Dn', '--cluster_dataSet_name', type=str, default='syn10k', help='聚类数据名称')
parser.add_argument('-Cn', '--number_of_cluster', type=int, default=10, help='聚类数量')
parser.add_argument('-Dnum', '--number_of_data', type=int, default=10000, help='数据数量')
parser.add_argument('-nf', '--number_of_features', type=int, default=2, help='特征数量')


# If randomly initialized
def init_FFCM_models(n_features, c, init_method='random'):
    init_cluster_centers = []
    n_features = n_features
    if init_method == 'random':
        for i in range(c):
            init_cluster_centers.append(list(np.random.rand(n_features)))
    if init_method == 'multi_normal':
        mean = [0] * n_features
        cov = np.identity(n_features)
        for i in range(0, c):
            init_cluster_centers.append(list(0.1 * np.random.multivariate_normal(mean, cov)))
    return init_cluster_centers

def nearest(point, cluster_centers):
    '''
    Calculate the minimum distance between point and cluster_centers
    :param point
    :param cluster_centers
    :return: Return the shortest distance between the point and the current cluster center
    '''
    min_dist = float("inf")
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        d = np.sqrt(np.sum(np.square(point - cluster_centers[i,])))
        if min_dist > d:
            min_dist = d
    return min_dist


def initcenters_with_kmeansPP(c_num, points, n_init=10):
    '''
        kmeans++
        :param points
        :param k: num of clusters
        :param n_init
        '''
    points = np.array(points)
    n, m = np.shape(points)
    # print('points', points)
    # print('m,n:', m, n)

    best_cluster_centers = None
    best_inertia = float("inf")

    for _ in range(n_init):
        cluster_centers = np.zeros((c_num, m))
        index = np.random.randint(0, n)
        cluster_centers[0,] = np.copy(points[index,])

        d = [0.0 for _ in range(n)]

        for i in range(1, c_num):
            sum_all = 0
            for j in range(n):
                d[j] = nearest(points[j,], cluster_centers[0:i, ])
                sum_all += d[j]
            sum_all *= random.random()
            for j, di in enumerate(d):
                if np.isinf(sum_all) == True:
                    sum_all = 1e06
                sum_all -= di
                if sum_all > 0:
                    continue
                cluster_centers[i] = np.copy(points[j,])
                break
        inertia = 0
        for point in points:
            inertia += nearest(point, cluster_centers)
        if inertia < best_inertia:
            best_inertia = inertia
            best_cluster_centers = cluster_centers

    return best_cluster_centers


def update_fuzzy_matrix(df, fuzzy_matrix, n_sample, c, m, cluster_centers):
    # Exponential term of the denominator
    order = float(2 / (m - 1))
    for i in range(n_sample):
        sample = list(df.iloc[i])
        # cal distances
        distances = [np.linalg.norm(np.array(list(map(operator.sub, sample, cluster_centers[j])))) \
                     for j in range(c)]
        for j in range(c):
            # Update the denominator of the formula
            denominator = sum([math.pow(float(distances[j] / distances[val]), order) for val in range(c)])
            fuzzy_matrix[i][j] = float(1 / denominator)
    return fuzzy_matrix


def init_fuzzy_matrix(n_sample, c):
    # shape = [n_sample, c]
    fuzzy_matrix = []
    for i in range(n_sample):
        random_list = [random.random() for i in range(c)]
        sum_of_random = sum(random_list)
        norm_random_list = [x / sum_of_random for x in random_list]
        one_of_random_index = norm_random_list.index(max(norm_random_list))
        for j in range(0, len(norm_random_list)):
            if (j == one_of_random_index):
                norm_random_list[j] = 1
            else:
                norm_random_list[j] = 0
        fuzzy_matrix.append(norm_random_list)
    return fuzzy_matrix


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def vali(name, resultCenters, is_IID):
    valSet = GetDataSet(name, is_IID)
    valDf = valSet.cluster_dataFrame

    # true labels for validate
    classLabels = valSet.fulllabels
    n_sample = len(valDf)
    c = args['number_of_cluster']
    m = args["fuzzy_index"]

    fuzzy_matrix = init_fuzzy_matrix(n_sample, c)
    resultFuzzymatrix = update_fuzzy_matrix(valDf, fuzzy_matrix, n_sample, c, m, resultCenters)
    resultLabels = []

    for i in range(0, n_sample):
        resultLabels.append(resultFuzzymatrix[i].index(max(resultFuzzymatrix[i])))

    sum_cluster_distance = 0
    valDf = np.array(valDf)
    min_cluster_center_distance = float("inf")
    for i in range(0, c):
        for j in range(0, n_sample):
            sum_cluster_distance = sum_cluster_distance + resultFuzzymatrix[j][i] ** m * sum(
                pow(valDf[j, :] - resultCenters[i, :], 2))

    for i in range(c - 1):
        for j in range(i + 1, c):
            cluster_center_distance = sum(pow(resultCenters[i, :] - resultCenters[j, :], 2))
            if cluster_center_distance < min_cluster_center_distance:
                min_cluster_center_distance = cluster_center_distance


    classLabels = np.array(classLabels)
    resultLabels = np.array(resultLabels)

    NMIScore = metrics.normalized_mutual_info_score(classLabels, resultLabels)
    ARIScore = metrics.adjusted_rand_score(classLabels, resultLabels)
    ACC = acc(classLabels, resultLabels)

    nonFed_result = np.load(f'./result/centers/{name}_centralized_centers.npy')

    gap = 0
    for j in range(0, c):
        A = nonFed_result[j]
        B = resultCenters[j]
        gap = gap + np.sqrt(sum(np.power((A - B), 2)))

    print('ACC :', ACC, 'NMI Score :', NMIScore, "ARI Score :", ARIScore,  "gap:", gap)
    return ACC, NMIScore, ARIScore, gap



if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__
    models = None
    roundNMI = float(0)
    m = args["fuzzy_index"]
    f_num = args['number_of_features']
    P = int(copy.deepcopy(args['num_of_clients']) * args["cfraction"])
    data_set_name = args['cluster_dataSet_name']
    local_epoch = args['epoch']
    current_iter = 0
    c = args['number_of_cluster']

    initcenter = init_FFCM_models(f_num, args['number_of_cluster'], init_method='random')

    # Initialize control variates
    server_control = np.zeros(shape=(c, f_num))
    client_control = np.zeros(shape=(c, f_num))

    # communicate start
    cumm = args["num_comm"]

    models = initcenter
    myClientsGroup1 = ClientsGroup(args["cluster_dataSet_name"], args['IID'], args['num_of_clients'],
                                   args['number_of_cluster'], models,
                                   local_epoch, current_iter, args["local_learning_rate"], client_control,
                                   server_control, m)

    myClients = myClientsGroup1.clients_set

    # co_init
    if args["init_method"] == "co_init":

        print("use co-init!")
        num_of_clients = args['num_of_clients']
        for ii in range(0, num_of_clients):
            myClient = myClients[ii]
            myClient.preInit()
            client_models = np.array(myClient.support_centers)
            if ii == 0:
                pre_init_centers = client_models
            else:
                pre_init_centers = np.concatenate((pre_init_centers, client_models), axis=0)

        global_init_centers = initcenters_with_kmeansPP(c, pre_init_centers, 10)
        print("global iniit centers:", global_init_centers)
        np.save(f'./result/SC_{data_set_name}_co_init_centers.npy', global_init_centers)

        for ii in range(0, num_of_clients):
            myClient = myClients[ii]
            myClient.models = global_init_centers
            myClient.init_models = global_init_centers

    ACC_list = []
    NMI_list = []
    ARI_list = []
    gap_list = []
    obj_list = []

    # start_1 = time.perf_counter()
    # sum_iter_time = 0
    for current_iter in range(0, cumm):

        print("The", current_iter, "round start：")
        print("--" * 50)
        temp = np.zeros(shape=(c, f_num))

        Beta = 1 / P
        temp_c = np.zeros(shape=(c, f_num))
        sample_list = random.sample(range(0, args['num_of_clients']), P)
        print('beta:', Beta)

        # sum_client_time = 0
        for i in range(0, P):
            # start_2 = time.perf_counter()
            delta_v = 0

            # random clients  sampling
            client = sample_list[i]
            myClient = myClients[client]

            # if full clients,you need to uncomment the following content
            # myClient = myClients[i]
            # print(myClient)

            models = np.array(models)

            if current_iter != 0:
                myClient.models = models
                myClient.init_models = models

            myClient.server_control = server_control

            myClient.localClustering()
            client_models = np.array(myClient.models)

            delta_v = client_models - models
            temp_c = temp_c + myClient.delta_c

            temp = temp + delta_v
            # print("--" * 50)
            # print(np.array(myClient.models))
            # print("--" * 50)
        #     end_2 = time.perf_counter()
        #     sum_client_time = sum_client_time + (end_2 - start_2)
        #     print("client ", i, "use  : %s Seconds " % (end_2 - start_2))
        # avg_client_time = sum_client_time / P
        # print("The average client computation time is: ", avg_client_time)

        delta_c = temp_c * Beta
        server_control = server_control + args["cfraction"] * delta_c

        delta_data = 0
        # use_trick = args['use_dency']
        server_step = args["Server_learning_rate"]

        delta_data = temp * Beta
        models = models + (delta_data * server_step)

        print("The", current_iter, "round：", "**" * 50)

        endCenters = models
        endCenters = endCenters[np.argsort(endCenters[:, -1]), :]
        end_1 = time.perf_counter()

        # sum_iter_time = sum_iter_time + (end_1 - start_1)
        print('endCenters', endCenters)

        tempArray = np.array(endCenters)
        roundACC,roundNMI,roundARI,roundGAP = vali(args['cluster_dataSet_name'], endCenters, args['IID'])

        # cacul_Obj:
        global_obj = 0
        for ii in range(0, args['num_of_clients']):
            sam = myClients[ii]
            temp_obj = sam.get_obj(c, endCenters)
            # print(ii, 'th obj:', temp_obj)
            global_obj = global_obj + temp_obj
        obj = global_obj
        print('global obj', obj)

        # save
        ACC_list.append(roundACC)
        NMI_list.append(roundNMI)
        ARI_list.append(roundARI)
        gap_list.append(roundGAP)
        obj_list.append(obj)

    # save as npy
    ACC_array = np.array(ACC_list)
    NMI_array = np.array(NMI_list)
    ARI_array = np.array(ARI_list)
    GAP_array = np.array(gap_list)
    OBJ_array = np.array(obj_list)
    is_IID = args['IID']

    # LOAD_NPY
    # ACC_array = np.load(f'./result/SC_{data_set_name}_{is_IID}_ACC_array.npy')
    # NMI_array = np.load(f'./result/SC_{data_set_name}_{is_IID}_NMI_array.npy')
    # ARI_array = np.load(f'./result/SC_{data_set_name}_{is_IID}_ARI_array.npy')
    # XB_array = np.load(f'./result/SC_{data_set_name}_{is_IID}_XB_array.npy')
    # # GAP = np.load(f'./result/AAAI2_{data_set_name}_{is_IID}_gap.npy')
    # OBJ_array = np.load(f'./result/SC_{data_set_name}_{is_IID}_Obj_array.npy')

    SC_ACC_MEAN = np.mean(ACC_array[100:500])
    SC_ACC_std = np.std(ACC_array[100:500])

    SC_NMI_MEAN = np.mean(NMI_array[100:500])
    SC_NMI_std = np.std(NMI_array[100:500])

    SC_ARI_MEAN = np.mean(ARI_array[100:500])
    SC_ARI_std = np.std(ARI_array[100:500])

    SC_GAP_MEAN = np.mean(GAP_array[100:500])
    SC_GAP_std = np.std(GAP_array[100:500])

    print('--' * 100)
    print('SC ACC mean:', SC_ACC_MEAN)
    print('SC ACC std:', SC_ACC_std)

    print('SC NMI mean:', SC_NMI_MEAN)
    print('SC NMI std:', SC_NMI_std)

    print('SC ARI mean:', SC_ARI_MEAN)
    print('SC ARI std:', SC_ARI_std)

    print('SC GAP mean:', SC_GAP_MEAN)
    print('SC GAP std:', SC_GAP_std)

    print('--' * 100)

    np.save(f'./result/SC_{data_set_name}_{is_IID}_ACC_array.npy', ACC_array)
    np.save(f'./result/SC_{data_set_name}_{is_IID}_NMI_array.npy', NMI_array)
    np.save(f'./result/SC_{data_set_name}_{is_IID}_ARI_array.npy', ARI_array)
    np.save(f'./result/SC_{data_set_name}_{is_IID}_GAP_array.npy', GAP_array)
    np.save(f'./result/SC_{data_set_name}_{is_IID}_Obj_array.npy', OBJ_array)

    print("final centers：")
    print(endCenters)







