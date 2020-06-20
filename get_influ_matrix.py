import numpy as np
import os
import argparse


def normalize_number(number):
    if (number > -100) and (number < 100):
        return number
    else:
        return 0


def normalize_array(input_array):
    for i in range(0, len(input_array)):
        input_array[i] = list(map(normalize_number, input_array[i]))
    return input_array


def concat_influence_and_save(folder_path, save_file):
    """
    :param folder_path: the path of folder which contain all influence files.
    :return:    the influence array.
    """
    file_num = len(list(os.listdir(folder_path)))
    influence = list()
    np.set_printoptions(precision=5)
    for i in range(0, file_num):
        array = np.loadtxt(folder_path + "/inf_of_a_test_point{}.txt".format(i))
        inf_list = list(map(normalize_number, array))
        influence.append(inf_list)
        if i % 100 == 0:
            print("{} have done".format(i))
    influence_array = np.array(influence).transpose()
    print(influence_array.shape)
    np.savetxt(fname=save_file, X=influence_array)
    print("saved influence_array. shape: ", influence_array.shape)


def select_the_error_index(folder_path):
    """select and print the error index of all index in the folder_path"""
    '''the average influence large than 100 will be seen as error index.'''
    error_index = list()
    file_num = len(list(os.listdir(folder_path)))
    for i in range(0, file_num):
        array = np.loadtxt(folder_path + "/inf_of_a_test_point{}.txt".format(i))
        if np.average(array) > 100:
            error_index.append(i)
    print(error_index)


def parse_args():
    parser = argparse.ArgumentParser(description="Get Influence Matrix.")
    parser.add_argument('--ResFolder', nargs='?', default="./Res_inf_SGC", help='Input results path')
    parser.add_argument('--SaveFile', nargs='?', default="./plot_data/sgc_all_inf.txt", help='IF file save path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ResFolder, SaveFile = args.ResFolder, args.SaveFile

    select_the_error_index(ResFolder)
    concat_influence_and_save(folder_path=ResFolder, save_file=SaveFile)

    # --ResFolder "./Res_inf_SGC" --SaveFile "./plot_data/sgc_all_inf.txt"
    # --ResFolder "./Res_inf_N2V" --SaveFile "./plot_data/n2v_all_inf.txt"
