from Mat_loader import *
import loguru
import argparse
from Cross_Model_Eval import *
def main():
    print('mat文件的命名规则是{binary_bits}-{modelname}-{dataset}.mat ')
    parser = argparse.ArgumentParser(description='Get Mat index')
    parser.add_argument('--file',default='./MAT/Flickr25k',help='path to mat')
    parser.add_argument('--dataset',default='flickr25k',help='name of dataset')
    parser.add_argument('--modelname',default='ours',help='modelname')
    parser.add_argument('--label',default=24,help='number of the class')

    parser.add_argument('--ifCrossModel',default=True,help='是否是跨模态')
    parser.add_argument('--if_i2t',default=True,help='image to text or text to image')
    args = parser.parse_args()
    if args.ifCrossModel :
        get_index = Cross_Mat_index(filepath=args.file,
                                    dataset=args.dataset,
                                    modelname=args.modelname,
                                    i2t=args.if_i2t,
                                    t2i=(args.if_i2t == False))
        get_index.topK_recall()
        get_index.prcurve()
        get_index.topK_precision(num=args.label)
        get_index.phamming2(num=args.label)
        get_index.NDCG_1000()
    else :
        get_index = Mat_index(filepath=args.file,
                              dataset=args.dataset,
                              modelname=args.modelname)
        get_index.topK_recall()
        get_index.prcurve()
        get_index.topK_precision(num=args.label)
        get_index.phamming2(num=args.label)
        get_index.NDCG_1000()

if __name__ == '__main__':
    main()

