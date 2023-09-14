from Mat_loader import *
import loguru
import argparse
def main():
    print('mat文件的命名规则是{binary_bits}-{dataset}-{modelname}')
    parser = argparse.ArgumentParser(description='Get Mat index')
    parser.add_argument('--file',default='./MAT',help='path to mat')
    parser.add_argument('--dataset',default='cifar10',help='name of dataset')
    parser.add_argument('--modelname',default='VIT',help='modelname')
    parser.add_argument('--label',default=10,help='number of the class')
    args = parser.parse_args()
    get_index = Mat_index(filepath=args.file,dataset=args.dataset,modelname=args.modelname)
    get_index.topK_recall()
    get_index.prcurve()
    get_index.topK_precision(num=args.label)
    get_index.phamming2(num=args.label)
    get_index.NDCG_1000()
if __name__ == '__main__':
    main()

