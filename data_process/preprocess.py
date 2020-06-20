'''
@File  :preprocess.py
@Author:Morton
@Date  :2020/6/18  11:59
@Desc  :Get raw content and mention graph by using DataLoader class and some saving function.
'''
# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import networkx as nx

from my_utils import load_obj, dump_obj, parse_args
from data_process.dataloader import DataLoader
from data_process.doc2vec import gensim_models_doc2vec


def preprocess_data(data_args):
    data_dir = data_args.dir
    dump_file = data_args.dump_file
    bucket_size = data_args.bucket
    encoding = data_args.encoding
    celebrity_threshold = data_args.celebrity
    mindf = data_args.mindf
    builddata = data_args.builddata
    doc2vec_model_file = data_args.doc2vec_model_file
    if os.path.exists(dump_file):
        if not builddata:
            print('loading data from file : {}'.format(dump_file))
            data = load_obj(dump_file)
            return data

    dl = DataLoader(data_home=data_dir, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()  # 'user'        df_train          df_dev          df_test
    dl.assignClasses()  # 'lat', 'lon'  train_classes     dev_classes     test_class
    if not os.path.exists(doc2vec_model_file):
        print("save all content and train to get doc2vec features...")
        dl.get_raw_content_and_save(
            save_file_path=data_dir + "corpus/content_all.txt")  # save the all content into file.
        gensim_models_doc2vec(raw_file=data_dir + "corpus/content_all.txt", doc2vec_model_file=doc2vec_model_file)

    print('loading doc2vec features from file: {}'.format(doc2vec_model_file))
    dl.load_doc2vec_feature(doc2vec_model_file)  # 'text'        X_train           X_dev           X_test
    U_train, U_dev, U_test = dl.df_train.index.tolist(), dl.df_dev.index.tolist(), dl.df_test.index.tolist()

    dl.get_graph()
    X_train, X_dev, X_test = dl.X_train, dl.X_dev, dl.X_test
    Y_train, Y_dev, Y_test = dl.train_classes, dl.dev_classes, dl.test_classes

    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]

    classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    adj = nx.adjacency_matrix(dl.graph)
    print('adjacency matrix created.')

    edge_pair_file = data_dir + "edge/edge_pair.ungraph"
    if not os.path.exists(edge_pair_file):
        get_edge_pair_from_adj(adj, edge_pair_file)

    data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test,
            classLatMedian, classLonMedian, userLocation)
    dump_obj(data, dump_file)
    print('successfully dump data in {}'.format(str(dump_file)))
    return data


def get_edge_pair_from_adj(adj, edge_pair_file):
    print("get_edge_pair_from_adj ...")
    adj = adj.tocoo()
    row = adj.row
    col = adj.col
    all_node_num = adj.shape[0]
    edge_list = np.vstack((row, col))
    edge_list = edge_list.T.tolist()
    edge_list_new = list(filter(lambda e: e[0] <= e[1], edge_list))

    # add self-loop for orphan.
    nodes_set = set([e[0] for e in edge_list_new] + [e[1] for e in edge_list_new])
    all_nodes_set = set([node for node in range(0, all_node_num)])
    supp_set = all_nodes_set - nodes_set
    for node in supp_set:
        edge_list_new.append([node, node])
    print("len(edge_list_new):", len(edge_list_new))

    # write into the file.
    out_str = ""
    with open(edge_pair_file, 'w') as f:
        for index, edge in enumerate(edge_list_new):
            out_str += str(edge[0]) + "\t" + str(edge[1]) + '\n'
            if (index % 1000 == 0) or (index == len(edge_list_new) - 1):
                f.write(out_str)
                out_str = ""
    print("save edge_pair into: {}".format(edge_pair_file))


def main():
    args = parse_args(sys.argv[1:])
    raw_data = preprocess_data(args)

    print("done.")


main()
