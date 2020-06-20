# -*- coding: UTF-8 -*-
from my_utils import load_data_for_plot
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_cluster_influence(test_cluster_list, train_cluster_list, influence_array):
    """
    :param test_cluster_list:   test points which belong to the same cluster.
    :param train_cluster_list:  train points which are the same of the cluster which test points are.
    :param influence_array:     influence array of all training point (train_num * test_num).
    :return: [impact_in, impact_out]
    """
    cluster_in, cluster_out = list(), list()
    for train_id in range(0, influence_array.shape[0]):
        if train_id in train_cluster_list:
            cluster_in.append(influence_array[train_id])
        else:
            cluster_out.append(influence_array[train_id])

    def get_impact(cluster_in_or_out, test_cluster_list):
        cluster_array = np.array(cluster_in_or_out)
        avg_list = np.average(cluster_array, axis=0)
        impact_list = list()
        for test_id in test_cluster_list:  # select the impact from the test_cluster_list by indexing test point index.
            impact_list.append(avg_list[test_id])
        return np.average(impact_list)

    impact_in = get_impact(cluster_in, test_cluster_list)
    impact_out = get_impact(cluster_out, test_cluster_list)

    return [impact_in, impact_out]


def normalize_array(input_array):
    def normalize_number(number):
        if (number > -100) and (number < 100):
            return number
        else:
            return 0

    for i in range(0, len(input_array)):
        input_array[i] = list(map(normalize_number, input_array[i]))
    return input_array


def main(dump_file, inf_file, cluster_inf_file, cluster_coordinate_file):
    data = load_data_for_plot(dump_file, feature_norm='None')
    adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data

    cluster_num = len(set(labels))
    test_label_dict = {i: [] for i in range(cluster_num)}   # cluster_0 : [id_1, id_2, id_3 ... ] test
    train_label_dict = {i: [] for i in range(cluster_num)}  # cluster_0 : [id_1, id_2, id_3 ... ] train

    for i, value in enumerate(labels[idx_test]):
        test_label_dict[value].append(i)

    for i, value in enumerate(labels[idx_train]):
        train_label_dict[value].append(i)

    # get influence array of all training point (train_num * test_num)
    influence_array = np.loadtxt(inf_file)
    # influence_array = normalize_array(influence_array)
    print("load Done.")

    # impact of all cluster
    impact_dict = {i: [] for i in range(cluster_num)}  # cluster : [impact_in, impact_out]
    for i in range(0, cluster_num):
        test_cluster_list = test_label_dict[i]
        train_cluster_list = train_label_dict[i]
        data = get_cluster_influence(test_cluster_list, train_cluster_list, influence_array)
        impact_dict[i].append(data)

    with open(cluster_inf_file, 'wb') as f:
        pickle.dump(impact_dict, f)
    print("cluster_inf save Done.")

    dump_data = (classLatMedian, classLonMedian)
    with open(cluster_coordinate_file, 'wb') as f:
        pickle.dump(dump_data, f)
    print("cluster_coordinate save Done.")


def plot_cluster_influence_sgc(cluster_inf_file):
    with open(cluster_inf_file, 'rb') as f:
        impact_dict = pickle.load(f)

    x_list = list(impact_dict.keys())
    y_va = list(impact_dict.values())
    y_in, y_out = list(), list()
    for y in y_va:
        y_in.append(y[0][0])
        y_out.append(y[0][1])

    def normalize_number(number):
        if (number > -10) and (number < 10):
            return number
        else:
            return 0

    y_in = list(map(normalize_number, y_in))
    y_out = list(map(normalize_number, y_out))

    # plot data.
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('Index of cluster', fontsize=13)
    ax.set_ylabel('Avg. Influence', fontsize=13)
    ax.scatter(x_list, y_in, c='g', s=20, alpha=0.5, label='in-cluster')
    ax.scatter(x_list, y_out, c='r', s=20, alpha=0.5, label='out-cluster')
    # ax.set_ylim([-0.08, 0.08])
    # plt.title('different of influence between in_cluster and out_cluster.')
    plt.legend(loc='center left', fontsize=13)
    plt.xticks([0, 50, 100])  # x 轴刻度密度
    plt.yticks([0, 0.05, 0.1])  # y 轴刻度密度
    plt.tick_params(labelsize=13)  # 刻度字体大小
    # plt.savefig('sgc_cluster_inf.png')
    plt.show()


def plot_3D_cluster_influence_sgc(cluster_inf_file, cluster_coordinate_file):
    with open(cluster_inf_file, 'rb') as f:
        impact_dict = pickle.load(f)
    with open(cluster_coordinate_file, 'rb') as f:
        dump_data = pickle.load(f)

    y_va = list(impact_dict.values())
    y_in, y_out = list(), list()
    for y in y_va:
        y_in.append(y[0][0])
        y_out.append(y[0][1])

    classLatMedian, classLonMedian = dump_data
    lat_list = list(classLatMedian.values())
    lon_list = list(classLonMedian.values())

    # plot data.
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax.scatter(lat_list, lon_list, y_in, c='#3f1ce1', s=30, alpha=1, label='pos-sample', marker='o')
    ax.scatter(lat_list, lon_list, y_out, c='#ff3300', s=30, alpha=1, label='neg-sample', marker='^')
    ax.set_xlabel('Latitude', fontsize=20, labelpad=10)
    ax.set_ylabel('Longitude', fontsize=20, labelpad=10)
    ax.set_zlabel('Avg. Influence', fontsize=20, labelpad=10)
    ax.set_xticks([30, 35, 40, 45])  # x 轴刻度密度
    ax.set_yticks([-70, -80, -90, -100, -110, -120])  # y 轴刻度密度
    ax.set_zticks([0.02, 0.04, 0.06, 0.08, 0.1])  # z 轴刻度密度
    plt.tick_params(labelsize=13)  # 刻度字体大小
    # plt.legend(loc=(-0.1, 0.5), fontsize=13, ncol=1, labelspacing=0.1)
    plt.subplots_adjust(left=0.0, right=0.9, top=1, bottom=0.1)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig('../pic/sgc_cluster_3D.png')
    plt.show()

    # -12, 19


def plot_cluster_influence_n2v(cluster_inf_file):
    with open(cluster_inf_file, 'rb') as f:
        impact_dict = pickle.load(f)

    x_list = list(impact_dict.keys())
    y_va = list(impact_dict.values())
    y_in, y_out = list(), list()
    for y in y_va:
        y_in.append(y[0][0])
        y_out.append(y[0][1])

    def normalize_number(number):
        if (number > -10) and (number < 10):
            return number
        else:
            return 0

    y_in = list(map(normalize_number, y_in))
    y_out = list(map(normalize_number, y_out))

    # plot data.
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('Index of cluster', fontsize=13)
    ax.set_ylabel('Avg. Influence', fontsize=13)
    ax.scatter(x_list, y_in, c='g', s=20, alpha=0.5, label='in-cluster')
    ax.scatter(x_list, y_out, c='r', s=20, alpha=0.5, label='out-cluster')
    # ax.set_ylim([-0.08, 0.08])
    # plt.title('different of influence between in_cluster and out_cluster.')
    plt.legend(loc='center left', fontsize=13)
    plt.xticks([0, 50, 100])			            # x 轴刻度密度
    plt.yticks([0, 0.15, 0.30, 0.45, 0.60])			# y 轴刻度密度
    plt.tick_params(labelsize=13)		            # 刻度字体大小
    # plt.savefig('n2v_cluster_inf.png')
    plt.show()


def plot_3D_cluster_influence_n2v(cluster_inf_file, cluster_coordinate_file):
    with open(cluster_inf_file, 'rb') as f:
        impact_dict = pickle.load(f)
    with open(cluster_coordinate_file, 'rb') as f:
        dump_data = pickle.load(f)

    y_va = list(impact_dict.values())
    y_in, y_out = list(), list()
    for y in y_va:
        y_in.append(y[0][0])
        y_out.append(y[0][1])

    classLatMedian, classLonMedian = dump_data
    lat_list = list(classLatMedian.values())
    lon_list = list(classLonMedian.values())

    # plot data.
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    plt.rcParams['savefig.dpi'] = 1000       # 图片像素
    plt.rcParams['figure.dpi'] = 1000        # 分辨率
    ax.scatter(lat_list, lon_list, y_in, c='#3f1ce1', s=30, alpha=1, label='pos-sample', marker='o')
    ax.scatter(lat_list, lon_list, y_out, c='#ff3300', s=30, alpha=1, label='neg-sample', marker='^')

    ax.set_xlabel('Latitude', fontsize=20, labelpad=10)
    ax.set_ylabel('Longitude', fontsize=20, labelpad=10)
    ax.set_zlabel('Avg. Influence', fontsize=20, labelpad=10)
    ax.set_xticks([30, 35, 40, 45])			            # x 轴刻度密度
    ax.set_yticks([-70, -80, -90, -100, -110, -120])    # y 轴刻度密度
    ax.set_zticks([0, 0.15, 0.30, 0.45, 0.60])     	    # z 轴刻度密度
    plt.tick_params(labelsize=13)                       # 刻度字体大小
    # plt.legend(loc=(-0.1, 0.5), fontsize=13, ncol=1, labelspacing=0.1)
    plt.subplots_adjust(left=0.0, right=0.9, top=1, bottom=0.1)
    # plt.tight_layout(rect=(0, 0, 1, 1))

    plt.savefig('../pic/n2v_cluster_3D.png')
    plt.show()

    # -12, 19


if __name__ == '__main__':
    """ For SGC model. """
    dump_file = "../dataset_cmu/dump_doc_dim_512.pkl"
    inf_file = "../plot_data/sgc_all_inf.txt"
    cluster_inf_file = "../plot_data/sgc_cluster_inf.dump"
    cluster_coordinate_file = "../plot_data/cluster_coordinate.dump"

    # main(dump_file, inf_file, cluster_inf_file, cluster_coordinate_file)
    # plot_cluster_influence_sgc(cluster_inf_file)
    plot_3D_cluster_influence_sgc(cluster_inf_file, cluster_coordinate_file)

    """ For MLP model. """
    # dump_file = "../dataset_cmu/dump_doc_dim_512.pkl"
    # inf_file = "../plot_data/n2v_all_inf.txt"
    # cluster_inf_file = "../plot_data/n2v_cluster_inf.dump"
    # cluster_coordinate_file = "../plot_data/cluster_coordinate.dump"

    # main(dump_file, inf_file, cluster_inf_file, cluster_coordinate_file)
    # plot_cluster_influence_n2v(cluster_inf_file)
    # plot_3D_cluster_influence_n2v(cluster_inf_file, cluster_coordinate_file)
