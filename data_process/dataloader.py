'''
@File  :dataloader.py
@Author:Morton
@Date  :2020/6/18  16:04
@Desc  :The basic loading function to extract raw content and mention graph information from raw data "user_info.xxx.gz".
'''
# -*- coding:utf-8 -*-
import os
import re
import csv
import kdtree
import gensim
import numpy as np
import pandas as pd
import networkx as nx
from haversine import haversine
from collections import defaultdict, OrderedDict
from sklearn.neighbors import NearestNeighbors


class DataLoader:
    def __init__(self, data_home, bucket_size=50, encoding='utf-8', celebrity_threshold=10, one_hot_labels=False,
                 mindf=10, maxdf=0.2, norm='l2', idf=True, btf=True, tokenizer=None, subtf=False, stops=None,
                 token_pattern=r'(?u)(?<![#@])\b\w\w+\b', vocab=None):
        self.data_home = data_home
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops if stops else 'english'
        self.token_pattern = r'(?u)(?<![#@|,.-_+^……$%&*(); :`，。？、：；;《》{}“”~#￥])\b\w\w+\b'
        self.vocab = vocab

    def load_data(self):
        print('loading the dataset from: {}'.format(self.data_home))
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                             quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)

        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)

        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)

        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test

    def get_graph(self):
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        print('start adding the train graph')
        externalNum = 0
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(user_id, id)
        print('start adding the dev graph')
        externalNum = 0
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('start adding the test graph')
        externalNum = 0
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(g), nx.number_of_edges(g)))

        celebrities = []
        for i in range(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)

        print('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
        print('projecting the graph')
        projected_g = self.efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)))
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        self.graph = projected_g

    def efficient_collaboration_weighted_projected_graph2(self, B, nodes):
        # B:        the whole graph including known nodes and mentioned nodes   --large graph
        # nodes:    the node_id of known nodes                                  --small graph node
        nodes = set(nodes)
        G = nx.Graph()
        G.add_nodes_from(nodes)
        all_nodes = set(B.nodes())
        for m in all_nodes:
            nbrs = B[m]
            target_nbrs = [t for t in nbrs if t in nodes]
            # add edge between known nodesA(m) and known nodesB(n)
            if m in nodes:
                for n in target_nbrs:
                    if m < n:
                        if not G.has_edge(m, n):
                            # Morton added for exclude the long edges
                            G.add_edge(m, n)
            # add edge between known n1 and known n2,
            # just because n1 and n2 have relation to m, why ? ? ? Yes, it's right.
            for n1 in target_nbrs:
                for n2 in target_nbrs:
                    if n1 < n2:
                        if not G.has_edge(n1, n2):
                            G.add_edge(n1, n2)
        return G

    def get_raw_content_and_save(self, save_file_path):
        # Morton add for save the raw content data into files.
        if os.path.exists(save_file_path):
            print("content already saved.")
            return None
        data = list(self.df_train.text.values) + list(self.df_dev.text.values) + list(self.df_test.text.values)
        file = open(save_file_path, 'w', encoding='utf-8')
        for i in range(len(data)):
            file.write(str(data[i]) + '\n')
        file.close()
        print("content saved in {}".format(save_file_path))

    def load_doc2vec_feature(self, doc2vec_model_file):
        """
            doc2vec_model_file: the file that including all doc2vec features of the raw content.
        """
        # load model
        model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model_file)

        # train data features
        feature_list = list()
        index_l = 0
        index_r = len(self.df_train.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_train = np.array(feature_list)

        # dev data features
        feature_list = list()
        index_l = len(self.df_train.text)
        index_r = len(self.df_train.text) + len(self.df_dev.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_dev = np.array(feature_list)

        # test data features
        feature_list = list()
        index_l = len(self.df_train.text) + len(self.df_dev.text)
        index_r = len(self.df_train.text) + len(self.df_dev.text) + len(self.df_test.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_test = np.array(feature_list)

        print("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        print("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        print("test        n_samples: %d, n_features: %d" % self.X_test.shape)

    def assignClasses(self):
        """
            get labels of all samples. label == index number of cluster.
        """
        clusterer = kdtree.KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()
        cluster_points = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
        print('# the number of clusterer labels is: %d' % len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points])
            self.cluster_median[cluster] = (median_lat, median_lon)
        dev_locs = self.df_dev[['lat', 'lon']].values
        test_locs = self.df_test[['lat', 'lon']].values
        nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr.fit(np.array(list(self.cluster_median.values())))
        self.dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]

        self.train_classes = clusters

        if self.one_hot_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros((len(self.train_classes), num_labels), dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros((len(self.dev_classes), num_labels), dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros((len(self.test_classes), num_labels), dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test
