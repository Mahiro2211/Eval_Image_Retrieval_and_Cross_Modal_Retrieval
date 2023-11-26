from Utils.evaluate import *
import scipy.io as scio
from loguru import logger
from Utils import PR_Curve as PR
from Utils.NDCG import cal_NDCG
import glob

class Mat_index():
    def __init__(self , dataset , filepath ,color='blue' , modelname='DSH' ,K=[1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
        print('mat文件的命名规则是{binary_bits}-{modelname}-{dataset}')
        self.modelname = modelname
        self.dataset = dataset
        self.filepath = filepath
        self.color = color
        self.bits = 16
        self.K = K
        self.baseroot = './Result'

    def load_mat(self): # 迭代器
        file_list = glob.glob(os.path.join(self.filepath,f'*{self.dataset}*.mat'))
        print(file_list)
        file_basename = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
        mat_dict = {i : scio.loadmat(file_list[i]) for i in range(len(file_list))}
        for i in range(len(file_list)):
            print(f'Processing the {i + 1} th file')
            qb , ql = mat_dict[i]['q_img'] , mat_dict[i]['q_l']
            rb , rl = mat_dict[i]['r_img'] , mat_dict[i]['r_l']
            yield qb,rb,ql,rl

    def prcurve(self):
        iter_mat = iter(self.load_mat())
        while True :
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                qb = torch.from_numpy(qb)
                rb = torch.from_numpy(rb)
                ql = torch.from_numpy(ql)
                rl = torch.from_numpy(rl)
                p, r = PR.pr_curve(qb, rb, ql, rl)
                save_csv('saved_index_pr',p,r,f'{self.bits}_{self.dataset}_Precison',f'{self.bits}_{self.dataset}_Recall')
            except StopIteration :
                print("Finish computing PRcurve")
                break
    def topK_recall(self):
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb, rb, ql, rl = next(iter_mat)
                self.bits = qb.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                recallK = []
                for i in self.K:
                    print(i)
                    _,a,_= mean_average_precision_normal_optimized_topK(rb,rl,qb,ql,i)
                    print(f'{self.bits}_top{i}_recall is {a}')
                    recallK.append(a)
                save_csv('saved_TopK_recall', self.K, recallK, 'K', f'{self.bits}_{self.dataset}_Recall')
            except StopIteration:
                break

    def topK_precision(self,num=10):
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb, rb, ql, rl = next(iter_mat)
                self.bits = qb.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                K = self.K
                P = []
                for i in self.K:
                    print(i)
                    a,_,_ = mean_average_precision_normal_optimized_topK(rb, rl, qb, ql, i)
                    print(f'{self.bits}_top{i}_precision is {a}')
                    P.append(a)
                save_csv('saved_topK_precision' , K , P , 'K' , f'{self.bits}_{self.dataset}_Precision')
            except StopIteration:
                break
    def NDCG_1000(self):
        "numpy_array"
        logger.add(f'the NDCG of {self.dataset} with {self.modelname}', format="{time} {level} {message}", level="INFO")
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                logger.info(f'{self.bits}_{self.dataset}_NDCG : {cal_NDCG(qb,rb,ql,rl,what=0,k=1000)}')
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
                qb , rb , ql , rl = next(iter_mat)
                rl[rl == -1] = 0
                ql[ql == -1] = 0
                self.bits = qb.shape[1]
                print(f'Calculating {self.bits} ......')
                precision, recall, map = get_precision_recall_by_Hamming_Radius(database_output=rb,
                                                                                database_labels=rl,
                                                                                query_output=qb,
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
                np.savetxt(f'./Result/PH@2_{self.dataset}_{self.modelname}',new_ph2,delimiter=' ')
                print('Finish computing PH@2')
                break






#%%

#%%
