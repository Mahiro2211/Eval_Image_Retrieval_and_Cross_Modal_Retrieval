'''
用于保存用于监督哈希绘制指标的算法库
分别为 ： TopK-Recall
        TopK-Precision
        PH@2
'''

import matplotlib
matplotlib.use('WebAgg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import os
import time

class LabelMatchs(object):
    def __init__(self, label_match_matrix):
        self.label_match_matrix = label_match_matrix
        self.all_sims = np.sum(label_match_matrix, axis=1)

def calc_label_match_matrix(database_labels, query_labels):
    return LabelMatchs(np.dot(query_labels, database_labels.T) > 0)

def partition_arg_topK(matrix, K, axis=0):
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def mean_average_precision_normal_optimized_topK(database_output, database_labels, query_output, query_labels, R,
                                                 verbose=0, label_matchs=None):
    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0

    label_matrix_time = -1
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)
        label_matrix_time = time.time() - tmp_time
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))

    query_num = query_output.shape[0]

    sim = -np.dot(query_output, database_output.T)
    start_time = time.time()
    topk_ids = partition_arg_topK(sim, R, axis=1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))

    column_index = np.arange(query_num)[:, None]
    imatchs = label_matchs.label_match_matrix[column_index, topk_ids]
    relevant_nums = np.sum(imatchs, axis=1)

    recX = relevant_nums.astype(float) / label_matchs.all_sims
    precX = relevant_nums.astype(float) / R

    Lxs = np.cumsum(imatchs, axis=1)
    Pxs = Lxs.astype(float) / np.arange(1, R + 1, 1)
    APxs = np.sum(Pxs * imatchs, axis=1)[relevant_nums > 0] / relevant_nums[relevant_nums > 0]
    meanAPxs = np.sum(APxs) / query_num
    if verbose > 0:
        print("MAP: %f" % meanAPxs)
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))
    print("total time(no label matrix): {:.3f}".format(time.time() - start_time))
    if label_matrix_time > 0:
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))
    return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), meanAPxs

def save_csv(filename , v1 , v2 ,v1_name , v2_name):
    final = '.csv'
    os.makedirs(os.path.join("./Result", filename), exist_ok=True)
    np.savetxt(os.path.join("./Result", filename, v1_name + final), v1)
    np.savetxt(os.path.join("./Result", filename, v2_name + final), v2)

def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    "投入的标签都需要进行onehot编码 ， 并且是numpy数组"
    # signed_query_output = np.sign(query_output) # -1 0 1 处理
    # signed_database_output = np.sign(database_output)
    signed_query_output = query_output
    signed_database_output = database_output
    bit_n = signed_query_output.shape[1]

    ips = np.dot(signed_query_output, signed_database_output.T)
    ips = (bit_n - ips) / 2

    start_time = time.time()
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))

    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    zero_count = 0
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(float(match_num) / all_num)
            matchX.append(match_num)
            allX.append(all_num)
            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(float(match_num) / all_sim_num)
            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(float(match_num) / all_num)

        else:
            print('zero: %d, no return' % zero_count)
            zero_count += 1
            precX.append(float(0.0))
            recX.append(float(0.0))
            mAPX.append(float(0.0))
            matchX.append(0.0)
            allX.append(0.0)
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))


