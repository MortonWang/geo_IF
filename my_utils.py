'''
@File  :my_utils.py
@Author:Morton
@Date  :2020/6/18  16:26
@Desc  :some utils which will been used in my project, including: dump and load function,
        normalized function and parse_args()
'''
# -*- coding:utf-8 -*-
import gzip
import pickle
import argparse
import numpy as np
import scipy.sparse as ssp


def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    with gzip.open(filename, 'wb') as fout:
        serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin)
    return obj


def aug_normalized_adjacency(adj):
    print("aug_normalized_adjacency of SGC.")
    adj = adj + ssp.eye(adj.shape[0])
    adj = ssp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).toarray()


def feature_normalization(dt, mode='None'):
    '''
        normalization according to row, each row represent a feature
    '''
    print("features normalization: {}".format(mode))
    if mode == "Standard":
        mean_num = np.mean(dt, axis=0)
        sigma = np.std(dt, axis=0)
        out = (dt - mean_num) / sigma
    if mode == "Mean":
        mean_num = np.mean(dt, axis=0)
        max_num = np.max(dt, axis=0)
        min_num = np.min(dt, axis=0)
        out = (dt - mean_num) / (max_num - min_num)
    if mode == "None":
        out = dt
    return out


def load_data_for_SGC(dump_file, feature_norm='None', degree=3):
    data = load_obj(dump_file)
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
    classLatMedian, classLonMedian, userLocation = data

    adj = aug_normalized_adjacency(adj)
    features = np.vstack((X_train, X_dev, X_test))  # only for dec2vec_feature
    features = feature_normalization(features, mode=feature_norm)
    print("features shape: {} , normalization:{}".format(features.shape, feature_norm))

    for i in range(0, degree):
        features = np.matmul(adj, features)

    labels = np.hstack((Y_train, Y_dev, Y_test))

    '''get index of train val and test'''
    len_train, len_val, len_test = int(X_train.shape[0]), int(X_dev.shape[0]), int(X_test.shape[0])
    idx_train, idx_val, idx_test = range(len_train), range(len_train, len_train + len_val), range(len_train + len_val,
                                                                                                  len_train + len_val + len_test)

    data = (features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,
            userLocation)
    return data


def load_data_for_N2V(dump_file, edge_file, feature_norm='None'):
    data = load_obj(dump_file)
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
    classLatMedian, classLonMedian, userLocation = data

    edge_emb = np.load(edge_file)
    print("edge_emb shape: {}".format(edge_emb.shape))

    features = np.vstack((X_train, X_dev, X_test))  # only for dec2vec_feature
    features = feature_normalization(features, mode=feature_norm)
    print("features shape: {} , normalization:{}".format(features.shape, feature_norm))

    labels = np.hstack((Y_train, Y_dev, Y_test))

    '''get index of train val and test'''
    len_train = int(X_train.shape[0])
    len_val = int(X_dev.shape[0])
    len_test = int(X_test.shape[0])
    idx_train = range(len_train)
    idx_val = range(len_train, len_train + len_val)
    idx_test = range(len_train + len_val, len_train + len_val + len_test)

    data = (features, labels, idx_train, idx_val, idx_test, U_train, U_dev,
            U_test, classLatMedian, classLonMedian, userLocation, edge_emb)
    return data


def load_data_for_plot(dump_file, feature_norm='None'):
    data = load_obj(dump_file)
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
    classLatMedian, classLonMedian, userLocation = data

    features = np.vstack((X_train, X_dev, X_test))  # only for dec2vec_feature
    features = feature_normalization(features, mode=feature_norm)
    print("features shape: {} , normalization:{}".format(features.shape, feature_norm))

    labels = np.hstack((Y_train, Y_dev, Y_test))

    '''get index of train val and test'''
    len_train = int(X_train.shape[0])
    len_val = int(X_dev.shape[0])
    len_test = int(X_test.shape[0])
    idx_train = range(len_train)
    idx_val = range(len_train, len_train + len_val)
    idx_test = range(len_train + len_val, len_train + len_val + len_test)

    data = (adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev,
            U_test, classLatMedian, classLonMedian, userLocation)
    return data


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """
    parser = argparse.ArgumentParser()

    # data_args: control the data loading
    parser.add_argument('-dir', metavar='str', help='the detail directory of dataset',
                        type=str, default='../dataset_cmu/')
    parser.add_argument('-dump_file', metavar='str', help='the dir to load file include name',
                        type=str, default='../dataset_cmu/dump_doc_dim_512.pkl')
    parser.add_argument('-feature_norm', type=str, choices=['None', 'Standard', 'Mean'], default='None')

    parser.add_argument('-doc2vec_model_file', metavar='str', help='the dir to load doc2vec_model_file .bin',
                        type=str, default="../dataset_cmu/corpus/model_dim_512_epoch_40.bin")

    # For cmu -bucket 50 -celebrity 5
    # For na -bucket 2400 -celebrity 15
    # For world -bucket 2400 -celebrity 5
    # parser.add_argument('-edge_dis_file', metavar='str', help='the dir to load all edges distance file',
    # type=str, default='./my_assets/no_ues_now/edge_dis.pkl')
    parser.add_argument('-bucket', metavar='int', help='discretisation bucket size', type=int, default=50)
    parser.add_argument('-mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument('-encoding', metavar='str', help='Data Encoding (e.g.latin1, utf-8)', type=str,
                        default='latin1')
    parser.add_argument('-celebrity', metavar='int', help='celebrity threshold', type=int, default=5)
    parser.add_argument('-builddata', action='store_true', help='if true do not recreated dumped data', default=False)

    # process_args: control the data preprocess
    parser.add_argument('-degree', type=int, help='degree of the approximation.', default=4)
    parser.add_argument('-normalization', type=str, help='Normalization method for the adjacency matrix.',
                        choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN', 'AugNormAdj', 'NormAdj',
                                 'RWalk', 'AugRWalk', 'NoNorm'], default='AugNormAdj')

    # model_args: the hyper-parameter of SGC model
    parser.add_argument('-model', type=str, help='model to use.', default="SGC")
    parser.add_argument('-usecuda', action='store_true', help='Use CUDA for training.', default=True)
    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-epochs', type=int, help='Number of epochs to train.', default=30000)
    parser.add_argument('-lr', type=float, help='Initial learning rate.', default=0.001)
    parser.add_argument('-weight_decay', type=float, help='Weight decay (L2 loss on parameters).', default=5e-7)
    parser.add_argument('-patience', help='max iter for early stopping', type=int, default=30)
    parser.add_argument('-batch', metavar='int', help='SGD batch size', type=int, default=1024)

    args = parser.parse_args(argv)
    return args
