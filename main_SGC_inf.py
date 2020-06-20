'''
@File  :main_SGC_inf.py
@Author:Morton
@Date  :2020/6/18  16:18
@Desc  :
'''
# -*- coding: UTF-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_ncg

from my_utils import load_data_for_SGC
from hessians import hessian_vector_product


def add_layer(input_data, in_size, out_size, act_func=None, name=None):
    weights = tf.Variable(tf.random.normal([in_size, out_size]), name=name + '_weight')
    biases = tf.Variable(tf.random.normal([1, out_size]) + 0.001, name=name + '_biases')
    result = tf.matmul(input_data, weights) + biases
    if act_func is None:
        outputs = result
    else:
        outputs = act_func(result)
    return outputs


# def geo_eval(y_pred, U_ture, classLatMedian, classLonMedian, userLocation):
#
#     assert len(y_pred) == len(U_ture), "#preds: %d, #users: %d" % (len(y_pred), len(U_ture))
#
#     distances = []
#     latlon_pred = []
#     latlon_true = []
#     for i in range(0, len(y_pred)):
#         user = U_ture[i]
#         location = userLocation[user].split(',')
#         lat, lon = float(location[0]), float(location[1])
#         latlon_true.append([lat, lon])
#         prediction = str(y_pred[i])
#         lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
#         latlon_pred.append([lat_pred, lon_pred, y_pred[i]])
#         distance = haversine((lat, lon), (lat_pred, lon_pred))
#         distances.append(distance)
#
#     acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
#     # return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred
#     return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


""" load data for SGC model """
dump_file = "./dataset_cmu/dump_doc_dim_512.pkl"
data = load_data_for_SGC(dump_file, feature_norm='None')
features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data

""" SGC + influence by using tensorflow.  """
learning_rate = 0.01
graph_emb_size = 128
content_emb_size = 512
class_num = 129
training_epochs = 100
display_epoch = 10
patience = 10

x_input = tf.compat.v1.placeholder(tf.float32, [None, content_emb_size], name='contentEmbedding')
y_label = tf.compat.v1.placeholder(tf.int64, [None, class_num], name='LabelData')

hidden_1 = add_layer(x_input, content_emb_size, 512, act_func=tf.nn.relu, name='MLP_1')
output_x = add_layer(hidden_1, 512, class_num, act_func=None, name='MLP_2')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=output_x))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

pred = tf.argmax(output_x, axis=1)
acc = tf.equal(tf.argmax(output_x, 1), tf.argmax(y_label, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

'''calculate influence '''
all_params = tf.compat.v1.trainable_variables()
params = [tf.compat.v1.trainable_variables()[2]]  # only last layer's params
gradients = tf.gradients(loss, params)
v_placeholder = params
hessian_vector = hessian_vector_product(loss, params, v_placeholder)
'''calculate influence '''

# Initialize the variables (i.e. assign their default value)
init = tf.compat.v1.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.compat.v1.train.Saver()


def get_influence(test_x, test_y):
    # Done --> predicted_loss_diffs == First step：S test(for test point which interested)、
    # Done --> Second step：I up,loss(calculate the effect of each training point)
    inverse_hvp = get_inverse_hvp_cg(get_test_grad_loss(test_x, test_y)[0])

    num_to_remove = len(idx_train)
    predicted_loss_diffs = list()
    for idx_to_remove in range(0, num_to_remove):
        single_train_feed_dict = fill_feed_dict_with_one_ex(idx_to_remove)
        train_grad_loss_val = sess.run(gradients, feed_dict=single_train_feed_dict)
        predicted_loss_diffs.append(np.dot(inverse_hvp, train_grad_loss_val[0].flatten()) / num_to_remove)
    return np.array(predicted_loss_diffs)


def get_test_grad_loss(test_x, test_y):
    return sess.run(gradients, {x_input: test_x, y_label: test_y})


def get_inverse_hvp_cg(v):
    fmin_loss_fn = get_fmin_loss_fn(v)
    fmin_grad_fn = get_fmin_grad_fn(v)

    fmin_results = fmin_ncg(
        f=fmin_loss_fn,
        x0=np.concatenate(v),
        fprime=fmin_grad_fn,  # gradient
        fhess_p=get_fmin_hvp,
        callback=None,
        avextol=1e-8,
        maxiter=20)

    return get_vec_to_list_fn()(fmin_results)


def get_fmin_loss_fn(v):
    def get_fmin_loss(x):
        hessian_vector_val = minibatch_hessian_vector_val(get_vec_to_list_fn()(x))

        return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)

    return get_fmin_loss


def get_fmin_grad_fn(v):
    def get_fmin_grad(x):
        hessian_vector_val = minibatch_hessian_vector_val(get_vec_to_list_fn()(x))

        return np.concatenate(hessian_vector_val) - np.concatenate(v)

    return get_fmin_grad


def minibatch_hessian_vector_val(v):
    feed_dict = fill_feed_dict_with_all_ex()
    # Can optimize this
    feed_dict = update_feed_dict_with_v_placeholder(feed_dict, v)
    hessian_vector_val = sess.run(hessian_vector, feed_dict=feed_dict)
    hessian_vector_val = np.reshape(hessian_vector_val,
                                    np.shape(hessian_vector_val[0])[0] * np.shape(hessian_vector_val[0])[1])
    return [hessian_vector_val]


def get_fmin_hvp(x, p):
    hessian_vector_val = minibatch_hessian_vector_val(get_vec_to_list_fn()(p))

    return np.concatenate(hessian_vector_val)


def fill_feed_dict_with_all_ex():
    feed_dict = {
        x_input: features[idx_train],
        y_label: get_one_hot(labels[idx_train])
    }
    return feed_dict


def fill_feed_dict_with_one_ex(target_idx):
    feed_dict = {
        x_input: [features[target_idx]],
        y_label: get_one_hot([labels[target_idx]])
    }
    return feed_dict


def update_feed_dict_with_v_placeholder(feed_dict, vec):
    for pl_block, vec_block in zip(v_placeholder, [np.reshape(vec, v_placeholder[0].get_shape())]):
        feed_dict[pl_block] = vec_block
    return feed_dict


def get_vec_to_list_fn():
    def vec_to_list(v):
        return v

    return vec_to_list


def get_one_hot(y_label):
    one_hot_index = np.arange(len(y_label)) * class_num + y_label
    one_hot = np.zeros((len(y_label), class_num))
    one_hot.flat[one_hot_index] = 1
    return one_hot


""" start running the framework ..."""
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 分配50%
tf_config.gpu_options.allow_growth = True  # 自适应
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""""for each test sample, calculate it's influence on each training sample, i.e. inf_of_a_test_point"""
features_test, labels_test = features[idx_test], labels[idx_test]
error_index = list()        # !!! store the error index which should rerun after.
for i in range(0, len(idx_test)):
    with tf.compat.v1.Session(config=tf_config) as sess:
        sess.run(init)
        try:
            inf_of_a_test_point = get_influence([features_test[i]], get_one_hot([labels_test[i]]))
        except Exception:
            error_index.append(i)
            print("-----------------------------------------There is a RuntimeWarning at index:", i)
            with open("./error_index.txt", 'a') as f:
                f.write("\nTime:" + str(time.asctime(time.localtime(time.time()))) + "\t\tError_at_index:" + str(i))
            continue
        else:
            np.savetxt("./Res_inf_SGC/inf_of_a_test_point{}.txt".format(i), inf_of_a_test_point)
            print("Time:", time.asctime(time.localtime(time.time())),
                  "has done ---------------------------- {}".format(i))

# show and save the whole error_index
error_index_str = "\n\nTime:" + str(time.asctime(time.localtime(time.time()))) + \
                  " \t\tModel:SGC \nAll_Error_index:" + str(error_index)
print(error_index_str)
with open("./error_index.txt", 'a') as f:
    f.write(error_index_str)
