# -*- coding: UTF-8 -*-
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from my_utils import load_data_for_plot
from haversine import haversine


def normalize_array(input_array):
    def normalize_number(number):
        if (number > -10) and (number < 10):
            return number
        else:
            return 0

    for i in range(0, len(input_array)):
        input_array[i] = list(map(normalize_number, input_array[i]))
    return input_array


def save_distance_array(dump_file, dis_arr_file):
    data = load_data_for_plot(dump_file, feature_norm='None')
    adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, \
    U_test, classLatMedian, classLonMedian, userLocation = data

    '''get distance matrix.'''
    distance_array = np.zeros((5685, 1895))
    for train_index in range(0, distance_array.shape[0]):
        user_train = U_train[train_index]
        location_train = userLocation[user_train].split(',')
        lat_train, lon_train = float(location_train[0]), float(location_train[1])
        for test_index in range(0, distance_array.shape[1]):
            user_test = U_test[test_index]
            location_test = userLocation[user_test].split(',')
            lat_test, lon_test = float(location_test[0]), float(location_test[1])
            distance = haversine((lat_train, lon_train), (lat_test, lon_test))
            distance_array[train_index][test_index] = distance
        if train_index % 100 == 0:
            print("distance_array : train_index = ", train_index)
    np.savetxt(fname=dis_arr_file, X=distance_array)
    print("save Done.")


def save_dis_and_influ_to_file(inf_file, dis_arr_file, inf_dis_file, grade_gap=1):
    influence_array = np.loadtxt(fname=inf_file)
    influence_array = normalize_array(influence_array)
    distance_array = np.loadtxt(fname=dis_arr_file)

    dis_max, dis_min = distance_array.max(), distance_array.min()
    grade_num = int(math.floor((dis_max - dis_min) / grade_gap))
    print('dis_min:{} \t\t dis_max:{} \t\t grade_gap:{}'.format(dis_min, dis_max, grade_gap))

    '''count influence by grade_gap of distance.'''
    grade_inf = {i: [0] for i in range(0, grade_num + 1)}
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance = distance_array[i][j]
            index = int(math.floor((distance - dis_min) / grade_gap))  # math.ceil(2.3)-->3 math.floor(2.3)-->2
            grade_inf[index].append(influence_array[i][j])
        if i % 1000 == 0:
            print("have done. index : ", i)

    '''get and save grade_inf_attribute.'''
    grade_inf_attribute = {i: [0] for i in range(0, grade_num + 1)}  # [distance, min_inf, max_inf, mean_inf]
    for key in range(0, len(grade_inf)):
        inf_list = grade_inf[key]
        distance, min_inf, max_inf, mean_inf = (key + 1) * grade_gap, min(inf_list), max(inf_list), np.average(inf_list)
        grade_inf_attribute[key] = [distance, min_inf, max_inf, mean_inf]
    with open(inf_dis_file, 'wb') as f:
        pickle.dump(grade_inf_attribute, f)
    print("save Done.")


def get_x_y_data(model_dist, dist_threshold=4000):
    x_data, y_data = list(), list()
    for key in range(0, len(model_dist)):
        if model_dist[key][0] > dist_threshold:
            break
        x_data.append(model_dist[key][0])
        y_data.append(model_dist[key][3])
    return x_data, y_data


def plot_inf_semilogx_dist(sgc_file, n2v_file):
    # load data.
    with open(sgc_file, 'rb') as f:
        sgc_dist = pickle.load(f)  # [distance, min_inf, max_inf, mean_inf]
    with open(n2v_file, 'rb') as f:
        n2v_dist = pickle.load(f)  # [distance, min_inf, max_inf, mean_inf]

    # extract the useful information.
    max_num, threshold_num = 1000000000, 4000
    x_data, y_sgc = get_x_y_data(sgc_dist, dist_threshold=threshold_num)
    _, y_n2v = get_x_y_data(n2v_dist, dist_threshold=threshold_num)

    # plot data
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['savefig.dpi'] = 1000      # 图片像素
    plt.rcParams['figure.dpi'] = 1000       # 分辨率

    ax.semilogx(x_data, y_n2v, label='MLP', color='y', linestyle='solid', linewidth=2)
    ax.semilogx(x_data, y_sgc, label='SGC', color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Distance (kilometer)', fontsize=13)
    ax.set_ylabel('Avg. Influence', fontsize=20)
    ax.set_yticks([0, 0.1, 0.2])        # y 轴刻度密度
    ax.tick_params(labelsize=13)        # 刻度字体大小
    fig.legend(loc="upper right", fontsize=16, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncol=1,
               columnspacing=0.1, labelspacing=0.2, markerscale=1, shadow=True, borderpad=0.2, handletextpad=0.2)
    fig.set_tight_layout(tight='rect')
    plt.savefig("../pic/inf_dist_log.png")
    plt.show()


if __name__ == '__main__':
    dump_file = "../dataset_cmu/dump_doc_dim_512.pkl"
    dis_arr_file = "../plot_data/distance_array.txt"
    # save_distance_array(dump_file, dis_arr_file)

    sgc_inf = "../plot_data/sgc_all_inf.txt"
    sgc_inf_dis_file = "../plot_data/sgc_grade_inf_gap1.txt"
    # save_dis_and_influ_to_file(sgc_inf, dis_arr_file, sgc_inf_dis_file, grade_gap=1)

    n2v_inf = "../plot_data/n2v_all_inf.txt"
    n2v_inf_dis_file = "../plot_data/n2v_grade_inf_gap1.txt"
    # save_dis_and_influ_to_file(n2v_inf, dis_arr_file, n2v_inf_dis_file, grade_gap=1)

    plot_inf_semilogx_dist(sgc_inf_dis_file, n2v_inf_dis_file)
