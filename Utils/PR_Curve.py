import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.spatial.distance import cdist
import torch
# from sklearn.metrics import precision_recall_curve


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit+1)
    R = torch.zeros(num_query, num_bit+1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit+1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r

    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask

    # plt.plot(R, P, linestyle="-", marker='D', color='blue')
    # plt.grid(True)
    # # plt.xlim(0, 1)
    # # plt.ylim(0, 1)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # # plt.legend()  # 加图例
    # plt.show()

    return P, R


# K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


#
# # data = h5py.File("E/PR/16-ours-flickr25k-i2t.mat")
# data = scio.loadmat('E:\\AAA-My\\哈希编码\\DCMH\\result\\PR-curve\\128-ours-iapr-t2i.mat')
# qb = torch.from_numpy(data['q_txt'])
# rb = torch.from_numpy(data['r_img'])
# ql = torch.from_numpy(data['q_l'])
# rl = torch.from_numpy(data['r_l'])
# p, r = pr_curve(qb, rb, ql, rl)
# p1 = p_topK(qb, rb, ql, rl, K)
# np.savetxt("E:\\AAA-My\\哈希编码\\DCMH\\result\\128-ours-iapr-t2i-Precision.csv", p.data.numpy())
# np.savetxt("E:\\AAA-My\\哈希编码\\DCMH\\result\\128-ours-iapr-t2i-Recall.csv", r.data.numpy())
# np.savetxt("E:\\AAA-My\\哈希编码\\DCMH\\result\\128-ours-iapr-t2i-topk.csv", p1.data.numpy())
# print(p)
# print(r)
# print(p1)