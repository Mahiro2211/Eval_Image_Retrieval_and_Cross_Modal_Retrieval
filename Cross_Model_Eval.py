from Mat_loader import Mat_index

from Utils.evaluate import *
import scipy.io as scio
from loguru import logger
from Utils import PR_Curve as PR
from Utils.NDCG import cal_NDCG
import glob
class Cross_Mat_index(Mat_index):
    def __init__(self, dataset , filepath,i2t,t2i ,color='blue' ,
                 modelname='DSH'
                 ,K=[1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                 ):
        # super().__init__()
        print('mat文件的命名规则是{binary_bits}-{modelname}-{dataset}')
        self.modelname = modelname
        self.dataset = dataset
        self.filepath = filepath
        self.color = color
        self.bits = 16
        self.K = K
        self.baseroot = './Result'
        self.i2t = i2t
        self.t2i = t2i

    def load_mat(self): # 迭代器
        test = 'i2t' if self.i2t else 't2i'
        file_list = glob.glob(os.path.join(self.filepath,f'*{self.modelname}*{self.dataset}*{test}*.mat'))
        print(file_list)
        file_basename = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
        # self.basename = file_basename
        mat_dict = {i : scio.loadmat(file_list[i]) for i in range(len(file_list))}
        self.name_dict = {i :  file_basename[i] for i in range(len(file_basename))}
        for i in range(len(file_list)):
            print(f'Processing the {i + 1} th file')

            # if 'i2t' in file_basename[i]:
            #     self.i2t = True

            qib , ql , qtb = mat_dict[i]['q_img'] , mat_dict[i]['q_l'] ,mat_dict[i]['q_txt']
            rib , rl , rtb= mat_dict[i]['r_img'] , mat_dict[i]['r_l'] , mat_dict[i]['r_txt']
            yield qib,rib,qtb,rtb,ql,rl

    def prcurve(self):
        iter_mat = iter(self.load_mat())
        while True :
            try:
                qib,rib,qtb,rtb,ql,rl  = next(iter_mat)

                self.bits = qib.shape[1]
                qib = torch.from_numpy(qib)
                qtb = torch.from_numpy(qtb)
                rib = torch.from_numpy(rib)
                rtb = torch.from_numpy(rtb)
                ql = torch.from_numpy(ql)
                rl = torch.from_numpy(rl)
                if self.i2t :
                    p, r = PR.pr_curve(qib, rtb, ql, rl)
                    save_csv('saved_I2T_index_pr', p, r, f'{self.bits}_{self.dataset}_Precison',
                             f'{self.bits}_{self.dataset}_Recall')

                if self.t2i  :
                    p, r = PR.pr_curve(qtb, rib, ql, rl)
                    save_csv('saved_T2I_index_pr', p, r, f'{self.bits}_{self.dataset}_Precison',
                             f'{self.bits}_{self.dataset}_Recall')

            except StopIteration :
                print("Finish computing PRcurve")
                break
    def topK_recall(self):
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qib, rib, qtb, rtb, ql, rl = next(iter_mat)
                self.bits = qib.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                recallK = []
                if self.i2t :
                    for i in self.K:
                        print(i)
                        _, a, _ = mean_average_precision_normal_optimized_topK(rtb, rl, qib, ql, i)
                        print(f'{self.bits}_top{i}_recall is {a}')
                        recallK.append(a)
                    save_csv('saved_I2T_TopK_recall', self.K, recallK, 'K',
                             f'{self.bits}_{self.dataset}_Recall')

                else :
                    for i in self.K:
                        print(i)
                        _, a, _ = mean_average_precision_normal_optimized_topK(rib, rl, qtb, ql, i)
                        print(f'{self.bits}_top{i}_recall is {a}')
                        recallK.append(a)
                    save_csv('saved_T2I_TopK_recall', self.K, recallK, 'K',
                             f'{self.bits}_{self.dataset}_Recall')

            except StopIteration:
                break

    def topK_precision(self,num=10):
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qib, rib, qtb, rtb, ql, rl = next(iter_mat)
                self.bits = qib.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                K = self.K
                P = []
                if self.i2t :
                    for i in self.K:
                        print(i)
                        a, _, _ = mean_average_precision_normal_optimized_topK(rtb, rl, qib, ql, i)
                        print(f'{self.bits}_top{i}_precision is {a}')
                        P.append(a)
                    save_csv('saved_I2T_topK_precision', K, P, 'K',
                             f'{self.bits}_{self.dataset}_Precision')
                else :
                    for i in self.K:
                        print(i)
                        a, _, _ = mean_average_precision_normal_optimized_topK(rib, rl, qtb, ql, i)
                        print(f'{self.bits}_top{i}_precision is {a}')
                        P.append(a)
                    save_csv('saved_T2I_topK_precision', K, P, 'K', f'{self.bits}_{self.dataset}_Precision')
            except StopIteration:
                break
    def NDCG_1000(self):
        "numpy_array"
        logger.add(f'the NDCG of {self.dataset} with {self.modelname}', format="{time} {level} {message}", level="INFO")
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qib, rib, qtb, rtb, ql, rl = next(iter_mat)
                self.bits = qib.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                if self.i2t:
                    logger.info(f'{self.bits}_{self.dataset}_I2T_NDCG : {cal_NDCG(qib, rtb, ql, rl, what=0, k=1000)}')
                else :
                    logger.info(f'{self.bits}_{self.dataset}_T2I_NDCG : {cal_NDCG(qtb, rib, ql, rl, what=0, k=1000)}')
            except StopIteration:
                break

    def phamming2(self , num=10):
        iter_mat = iter(self.load_mat())
        ph2 = np.zeros((5,2))
        start = 0
        recall_ph2 = np.array([])
        Map = np.array([])
        while True:
            try:
                qib, rib, qtb, rtb, ql, rl = next(iter_mat)
                rl[rl == -1] = 0
                ql[ql == -1] = 0
                self.bits = qib.shape[1]
                print(f'Calculating {self.bits} ......')
                if self.i2t:
                    precision, recall, map = get_precision_recall_by_Hamming_Radius(database_output=rtb,
                                                                                    database_labels=rl,
                                                                                    query_output=qib,
                                                                                    query_labels=ql)
                else:
                    precision, recall, map = get_precision_recall_by_Hamming_Radius(database_output=rib,
                                                                                    database_labels=rl,
                                                                                    query_output=qtb,
                                                                                    query_labels=ql)

                # ph2 = np.append(ph2 , [self.bits,precision],axis=1)
                ph2[start] = [self.bits,precision] 
                start = start + 1               

            except StopIteration:
                print(ph2)
                # exit()
                first_coloum = ph2[:,0]
                indice = np.argsort(first_coloum)
                
                new_ph2 = ph2[indice]
                print(new_ph2)
                if self.i2t:
                    np.savetxt(f'./Result/PH@2_i2t_{self.dataset}_{self.modelname}', new_ph2, delimiter=' ')
                else :
                    np.savetxt(f'./Result/PH@2_t2i_{self.dataset}_{self.modelname}', new_ph2, delimiter=' ')
                print('Finish computing PH@2')
                break
