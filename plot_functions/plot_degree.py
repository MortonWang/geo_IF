# -*- coding: UTF-8 -*-
from my_utils import load_data_for_plot
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_neighbor_of_index(adj, index_list, index_limit=5685):
    """
    :param adj:   the adjacent matrix of graph with csr_matrix format.
    :param index_list: the index of points which we want to get it's neighbors.
    :param index_limit: the max index limit(not equal), in order to only remain the training point.
    :return:
    """

    def normalize_number(number):
        return True if number < index_limit else False

    neighbor = list()
    for i in index_list:
        nei_index = np.argwhere(adj[i] == 1)
        nei_index = list(nei_index.transpose()[1])
        nei_index = list(filter(normalize_number, nei_index))  # 将邻居关系限制在 training 节点内
        neighbor.append(nei_index)
    return neighbor


def get_neighbor_impact(neighbor_list, influence_array, end_index):
    """
    :param neighbor_list:       the neighbors of test point.
    :param influence_array:     influence array of all training point (train_num * test_num).
    :param end_index:           the index of train point ended in all points.
    :return:                    [impact_1, impact_2, ... ]
    """
    influence_array = influence_array[0:end_index].transpose()  # only remain the training points.
    impact_all = list()
    for i, neighbor in enumerate(neighbor_list):
        impact_of_one_point = list()
        for nei in neighbor:
            impact_of_one_point.append(influence_array[i][nei])
        if impact_of_one_point:
            impact_all.append(np.average(impact_of_one_point))
        else:
            impact_all.append(0)

    return impact_all


def main(dump_file, degree_nei_file):
    data = load_data_for_plot(dump_file, feature_norm='None')
    adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data

    '''get first degree neighbor and second degree neighbor for test point.'''
    first_degree_neighbor = get_neighbor_of_index(adj, idx_test, index_limit=len(idx_train))
    second_degree_neighbor = list()
    for fir_nei in first_degree_neighbor:
        neighbor = get_neighbor_of_index(adj, fir_nei, index_limit=len(idx_train))
        all_nei = list()
        for nei in neighbor:
            all_nei.extend(nei)
        second_degree_neighbor.append(list(set(all_nei) - set(fir_nei)))  # 限制二阶邻居中不包含一阶邻居

    thrid_degree_neighbor = list()
    for i, sec_nei in enumerate(second_degree_neighbor):
        neighbor = get_neighbor_of_index(adj, sec_nei, index_limit=len(idx_train))
        all_nei = list()
        for nei in neighbor:
            all_nei.extend(nei)
        thrid_degree_neighbor.append(list(set(all_nei) - set(first_degree_neighbor[i]) -
                                          set(second_degree_neighbor[i])))  # 限制三阶邻居中不包含一 二阶邻居
    data = (first_degree_neighbor, second_degree_neighbor, thrid_degree_neighbor)
    with open(degree_nei_file, 'wb') as f:
        pickle.dump(data, f)
    print("save degree neighbor done.")


def get_hops3_influence(degree_nei_file, inf_file, hops_inf_file):
    with open(degree_nei_file, 'rb') as f:
        data = pickle.load(f)
    first_degree_neighbor, second_degree_neighbor, thrid_degree_neighbor = data

    '''get first impact and second impact of neighbors for test point.'''
    influence_array = np.loadtxt(inf_file)
    first_impact = get_neighbor_impact(first_degree_neighbor, influence_array, end_index=5685)
    second_impact = get_neighbor_impact(second_degree_neighbor, influence_array, end_index=5685)
    thrid_impact = get_neighbor_impact(thrid_degree_neighbor, influence_array, end_index=5685)

    '''save to files.'''
    data = (first_impact, second_impact, thrid_impact)
    with open(hops_inf_file, 'wb') as f:
        pickle.dump(data, f)
    print("hop3_inf save Done.")


def abs_scacle_list(y_list, weight):
    res = list()
    for i in y_list:
        res.append(abs(i) * weight + 4)
    return res


def plot_hop3_influence_sgc(hops_inf_file):
    with open(hops_inf_file, 'rb') as f:
        data = pickle.load(f)
    first_impact, second_impact, thrid_impact = data

    x_list = list(range(0, len(first_impact)))
    y_first = first_impact
    y_second = second_impact
    y_thrid = thrid_impact

    def normalize_number(number):
        if (number > -100) and (number < 100):
            return number
        else:
            return 0

    y_first = list(map(normalize_number, y_first))
    y_second = list(map(normalize_number, y_second))
    y_thrid = list(map(normalize_number, y_thrid))

    less_0_list = list()
    for i, influence in enumerate(y_first):
        if influence < 0:
            less_0_list.append(i)
    for i, influence in enumerate(y_second):
        if influence < 0:
            less_0_list.append(i)
    for i, influence in enumerate(y_thrid):
        if influence < 0:
            less_0_list.append(i)
    less_0_set = set(less_0_list)
    print("len(less_0_set):\t", len(less_0_set), "\n", less_0_set)

    '''My plot'''
    plt.figure()
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = plt.gca()
    ax.set_xlabel('Index of testing nodes', fontsize=20)
    ax.set_ylabel('Avg. Influence', fontsize=20)

    ax.scatter(x_list, y_first, c='#ff3333', s=8, alpha=1, label='1-hop')
    ax.scatter(x_list, y_second, c='#8cff66', s=8, alpha=1, label='2-hop')
    ax.scatter(x_list, y_thrid, c='#3333ff', s=8, alpha=0.6, label='3-hop')
    # plt.title('different of influence between in_cluster and out_cluster.')
    ax.set_xlim([-10, 1905])
    plt.xticks([0, 600, 1200, 1800])  # x 轴刻度密度
    plt.yticks([0, 0.1, 0.2])  # y 轴刻度密度
    plt.tick_params(labelsize=16)  # 刻度字体大小
    plt.legend(loc='upper left', fontsize=20, ncol=1, columnspacing=0.1, labelspacing=0.1,
               markerscale=2, shadow=True, borderpad=0.2, handletextpad=0.1)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig('../pic/sgc_hop3.png')
    plt.show()


def plot_hop3_influence_mlp(hops_inf_file):
    with open(hops_inf_file, 'rb') as f:
        data = pickle.load(f)
    first_impact, second_impact, thrid_impact = data

    x_list = list(range(0, len(first_impact)))
    y_first = first_impact
    y_second = second_impact
    y_thrid = thrid_impact

    def normalize_number(number):
        if (number > -10) and (number < 2):
            return number
        else:
            return 0

    y_first = list(map(normalize_number, y_first))
    y_second = list(map(normalize_number, y_second))
    y_thrid = list(map(normalize_number, y_thrid))

    less_0_list = list()
    for i, influence in enumerate(y_first):
        if influence < 0:
            less_0_list.append(i)
    for i, influence in enumerate(y_second):
        if influence < 0:
            less_0_list.append(i)
    for i, influence in enumerate(y_thrid):
        if influence < 0:
            less_0_list.append(i)
    less_0_set = set(less_0_list)
    print("len(less_0_set):\t", len(less_0_set), "\n", less_0_set)

    # plot the point.
    plt.figure()
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = plt.gca()
    ax.set_xlabel('Index of testing nodes', fontsize=20)
    ax.set_ylabel('Avg. Influence', fontsize=20)
    ax.scatter(x_list, y_first, c='#ff3333', s=8, alpha=1, label='1-hop')
    ax.scatter(x_list, y_second, c='#8cff66', s=8, alpha=1, label='2-hop')
    ax.scatter(x_list, y_thrid, c='#3333ff', s=8, alpha=0.6, label='3-hop')
    # plt.title('different of influence between in_cluster and out_cluster.')
    ax.set_xlim([-10, 1905])
    plt.xticks([0, 600, 1200, 1800])    # x 轴刻度密度
    plt.yticks([0, 1, 2])               # y 轴刻度密度
    plt.tick_params(labelsize=16)       # 刻度字体大小
    plt.legend(loc='upper left', fontsize=20, ncol=1, columnspacing=0.1, labelspacing=0.1,
               markerscale=2, shadow=True, borderpad=0.2, handletextpad=0.1)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig('../pic/n2v_hop3.png')
    plt.show()


if __name__ == '__main__':
    """ For SGC model. """
    # dump_file = "../dataset_cmu/dump_doc_dim_512.pkl"
    # degree_nei_file = "../plot_data/degree_neighbor.dump"
    # inf_file = "../plot_data/sgc_all_inf.txt"
    hops_inf_file = "../plot_data/sgc_hop3_inf.dump"

    # main(dump_file, degree_nei_file)
    # get_hops3_influence(degree_nei_file, inf_file, hops_inf_file)
    plot_hop3_influence_sgc(hops_inf_file)

    """ For MLP model. """
    # dump_file = "../dataset_cmu/dump_doc_dim_512.pkl"
    # degree_nei_file = "../plot_data/degree_neighbor.dump"
    # inf_file = "../plot_data/n2v_all_inf.txt"
    # hops_inf_file = "../plot_data/n2v_hop3_inf.dump"

    # main(dump_file, degree_nei_file)
    # get_hops3_influence(degree_nei_file, inf_file, hops_inf_file)
    # plot_hop3_influence_mlp(hops_inf_file)
