import numpy as np
from sklearn.preprocessing import normalize
import scipy.io as scio
import torch

def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def calc_hamming_dist(B1, B2):
    B1 = torch.from_numpy(B1)
    B2 = torch.from_numpy(B2)
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def sim_mat(label, label_2=None, sparse=False):
    if label_2 is None:
        label_2 = label
    if sparse:
        S = label[:, np.newaxis] == label_2[np.newaxis, :]
    else:
        S = np.dot(label, label_2.T) > 0
    return S.astype(label.dtype)


def cal_NDCG(qF, rF, qL, rL, what=1, k=-1, sparse=False):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T).astype(int)
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(k))
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(calc_hamming_dist(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n_query

#
# data = scio.loadmat('E:\\AAA-My\\哈希编码\\DCMH\\result\\PR-curve\\128-ours-iapr-t2i.mat')
# qb = torch.from_numpy(data['q_txt'])
# rb = torch.from_numpy(data['r_img'])
# ql = torch.from_numpy(data['q_l'])
# rl = torch.from_numpy(data['r_l'])
#
# # qb = np.load('E:\\CSQ-IDHN\\[IDHN]_nuswide_21_64bits_0.7814583756654898\\tst_binary.npy')
# # ql = np.load('E:\\CSQ-IDHN\\[IDHN]_nuswide_21_64bits_0.7814583756654898\\tst_label.npy')
# # rb = np.load('E:\\CSQ-IDHN\\[IDHN]_nuswide_21_64bits_0.7814583756654898\\trn_binary.npy')
# # rl = np.load('E:\\CSQ-IDHN\\[IDHN]_nuswide_21_64bits_0.7814583756654898\\trn_label.npy')
# # qb = torch.from_numpy(qb)
# # ql = torch.from_numpy(ql)
# # rb = torch.from_numpy(rb)
# # rl = torch.from_numpy(rl)
#
# x = NDCG(qb, rb, ql, rl, what=1, k=1000)
# print(x)