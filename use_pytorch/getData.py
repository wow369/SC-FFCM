import numpy as np
import pandas as pd
import gzip
import os
import operator
import math
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.cluster_dataFrame = None
        self.features = None
        self.is_IID = isIID
        self.fulllabels = None
        self.num_of_cluster = None

        if self.name == 'iris':
            self.irisDataSetConstruct(self.is_IID)
        elif self.name == 'syn10k':
            self.syn_data_10k(self.is_IID)

    def irisDataSetConstruct(self, isIID):
            data = load_iris()
            # dataset
            features = data['data']

            # label
            target = data['target']

            target = target[:, np.newaxis]
            target_names = data['target_names']
            # target_dicts = dict(zip(np.unique(target), target_names))
            # feature_names:['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'label']
            feature_names = data['feature_names']

            feature_names = data['feature_names'].copy()  # deepcopy(data['feature_names'])
            feature_names.append('label')
            df_full = pd.DataFrame(data=np.concatenate([features, target], axis=1),
                                   columns=feature_names)

            if isIID == 1:
                df_full =pd.read_csv('./data/iris_random.csv')

            columns = list(df_full.columns)
            class_labels = list(df_full[columns[-1]])
            features = columns[:len(columns)-1]

            df = df_full[features]

            # normalization
            ss = MinMaxScaler()
            df = ss.fit_transform(df)
            df = pd.DataFrame(df)

            self.cluster_dataFrame = df
            self.features = features
            self.fulllabels = class_labels
            self.num_of_cluster = 3


    def syn_data_10k(self, isIID):

        fulldata = pd.read_csv('./data/random_syn10k.csv')
        if isIID == 0:
            fulldata = pd.read_csv('./data/sort_syn10k.csv')

        columns = list(fulldata.columns)
        class_labels = list(fulldata[columns[-1]])
        features = columns[:len(columns) - 1]
        data_df = fulldata[features]
        df = data_df

        # ss = MinMaxScaler()
        # df = ss.fit_transform(data_df)
        # df = pd.DataFrame(df)
        # print('df', df[:5])

        self.cluster_dataFrame = df
        self.features = features
        self.fulllabels = class_labels
        self.num_of_cluster = 10


if __name__ == "__main__":
    'test data set'
    #irisDataSet是Object类型，获得的数据是DataFrame类型
    irisDataSet = GetDataSet('iris', 0)#.cluster_data # test NON-IID
    # print(irisDataSet)


