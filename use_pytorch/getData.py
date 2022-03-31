import numpy as np
import pandas as pd
import gzip
import os
import operator
import math
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.cluster_dataFrame = None
        self.features = None
        # self._index_in_train_epoch = 0
        self.is_IID = isIID


        if self.name == 'iris':
            self.irisDataSetConstruct(self.is_IID)
        elif self.name == 'housing'and self.is_IID == 1:
            self.housingDataSetConstruct(self.is_IID)
        elif self.name == 'housing'and self.is_IID == 0:
            self.housingDataSetConstruct_Noiid(self.is_IID)

    def irisDataSetConstruct(self, isIID):
        # if isIID:
            data = load_iris()
            # iris数据集的特征列(数据本身)
            features = data['data']

            # iris数据集的标签（分类）
            target = data['target']
            # 增加维度1，用于拼接
            target = target[:, np.newaxis]
            target_names = data['target_names']
            # target_dicts = dict(zip(np.unique(target), target_names))
            # feature_names:['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'label']
            feature_names = data['feature_names']

            # 浅拷贝，防止原地修改
            feature_names = data['feature_names'].copy()  # deepcopy(data['feature_names'])
            feature_names.append('label')
            df_full = pd.DataFrame(data=np.concatenate([features, target], axis=1),
                                   columns=feature_names)

            # 写入文件
            # df_full.to_csv(str(os.getcwd()) + '/data/iris_data.csv', index=None)

            columns = list(df_full.columns)
            features = columns[:len(columns)]

            # 标签列
            # class_labels = list(df_full[columns[-1]])

            df = df_full[features]

            self.cluster_dataFrame = df
            self.features = features
            # print(target_dicts, df_full, df)
            # return df_full, df, class_labels, target_dicts
        # else:
        #     pass

    def housingDataSetConstruct(self, isIID):
        h = pd.read_csv('./h.csv')
        # features = ['avg_number_of_rooms', 'RAD', 'house_value']
        features = ['avg_number_of_rooms', 'distance_to_employment_centers', 'house_value']

        df = h[features]

        # shuffle操作，独立同分布化
        # df = shuffle(df)

        self.cluster_dataFrame = df
        self.features = features

    def housingDataSetConstruct_Noiid(self, isIID):
        h = pd.read_csv('./h-non-iid.csv')
        features = ['avg_number_of_rooms', 'RAD', 'house_value']
        # features = ['avg_number_of_rooms', 'distance_to_employment_centers', 'house_value']

        df = h[features]
        self.cluster_dataFrame = df
        self.features = features


#     def mnistDataSetConstruct(self, isIID):
#         data_dir = r'.\data\MNIST'
#         # data_dir = r'./data/MNIST'
#         train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
#         train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
#         test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
#         test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
#
#         train_images = extract_images(train_images_path)
#         train_labels = extract_labels(train_labels_path)
#         test_images = extract_images(test_images_path)
#         test_labels = extract_labels(test_labels_path)
#
#         assert train_images.shape[0] == train_labels.shape[0]
#         assert test_images.shape[0] == test_labels.shape[0]
#
#         self.train_data_size = train_images.shape[0]
#         self.test_data_size = test_images.shape[0]
#
#         assert train_images.shape[3] == 1
#         assert test_images.shape[3] == 1
#         train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
#         test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
#
#         train_images = train_images.astype(np.float32)
#         train_images = np.multiply(train_images, 1.0 / 255.0)
#         test_images = test_images.astype(np.float32)
#         test_images = np.multiply(test_images, 1.0 / 255.0)
#
#         if isIID:
#             order = np.arange(self.train_data_size)
#             np.random.shuffle(order)
#             self.train_data = train_images[order]
#             self.train_label = train_labels[order]
#         else:
#             labels = np.argmax(train_labels, axis=1)
#             order = np.argsort(labels)
#             self.train_data = train_images[order]
#             self.train_label = train_labels[order]
#
#
#
#         self.test_data = test_images
#         self.test_label = test_labels
#
#
# def _read32(bytestream):
#     dt = np.dtype(np.uint32).newbyteorder('>')
#     return np.frombuffer(bytestream.read(4), dtype=dt)[0]
#
#
# def extract_images(filename):
#     """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         magic = _read32(bytestream)
#         if magic != 2051:
#             raise ValueError(
#                     'Invalid magic number %d in MNIST image file: %s' %
#                     (magic, filename))
#         num_images = _read32(bytestream)
#         rows = _read32(bytestream)
#         cols = _read32(bytestream)
#         buf = bytestream.read(rows * cols * num_images)
#         data = np.frombuffer(buf, dtype=np.uint8)
#         data = data.reshape(num_images, rows, cols, 1)
#         return data
#
#
# def dense_to_one_hot(labels_dense, num_classes=10):
#     """Convert class labels from scalars to one-hot vectors."""
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#     return labels_one_hot
#
#
# def extract_labels(filename):
#     """Extract the labels into a 1D uint8 numpy array [index]."""
#     print('Extracting', filename)
#     with gzip.open(filename) as bytestream:
#         magic = _read32(bytestream)
#         if magic != 2049:
#             raise ValueError(
#                     'Invalid magic number %d in MNIST label file: %s' %
#                     (magic, filename))
#         num_items = _read32(bytestream)
#         buf = bytestream.read(num_items)
#         labels = np.frombuffer(buf, dtype=np.uint8)
#         return dense_to_one_hot(labels)


if __name__ == "__main__":
    'test data set'
    #irisDataSet是Object类型，获得的数据是DataFrame类型
    irisDataSet = GetDataSet('housing', 1)#.cluster_data # test NON-IID
    # print(irisDataSet)


