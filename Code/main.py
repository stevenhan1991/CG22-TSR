from model import InterpolationNet, MaskNet, weights_init_kaiming, TSRVFD, RTSRVFD, Dis, TSRVFDwoSep
from train import trainMaskNet, trainInterpolationNet, inference, trainTSRVFD, trainGAN
import math
import os
import argparse
from DataPre import DataSet
import torch
from metrics import GetMetrics

parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "TSR-VFD"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of TSR-VFD')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--dataset', type=str, default='cylinder', metavar='N',
                    help='the data set we used for training TSR-VFD')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--init_channels', type=int, default=16, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--samples', type=int, default=40, metavar='N',
                    help='the samples we used for training TSR-VFD')
parser.add_argument('--interval', type=int, default=3, metavar='N',
                    help='interpolation step')
parser.add_argument('--loss', type=str, default='mse', metavar='N',
                    help='loss function')
parser.add_argument('--mode', type=str, default='TSR', metavar='N',
                    help='')
parser.add_argument('--data_path', type=str, default='/data/junhan/TSR-VFD/VectorData/', metavar='N',
                    help='the path where we read the vector data')
parser.add_argument('--model_path', type=str, default='/data/junhan/TSR-VFD/Exp/', metavar='N',
                    help='the path where we stored the saved model')
parser.add_argument('--result_path', type=str, default='/data/junhan/TSR-VFD/Result/', metavar='N',
                    help='the path where we stored the synthesized data')
parser.add_argument('--train', type=str, default='train', metavar='N',
                    help='')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def main():
    if not os.path.exists(args.model_path+args.dataset):
        os.mkdir(args.model_path+args.dataset)
    VectorData = DataSet(args)
    VectorData.ReadData()
    if args.mode == 'Interpolation':
        model = InterpolationNet(inc=2,outc=args.interval,init_channels=args.init_channels,num=1)
        if args.cuda:
            model.cuda()
        model.apply(weights_init_kaiming)
        trainInterpolationNet(model,args,VectorData)
    elif args.mode == 'TSR':
        model = TSRVFD(inc=2,outc=args.interval,init_channels=args.init_channels,num=1)
        if args.cuda:
            model.cuda()
        model.apply(weights_init_kaiming)
        trainTSRVFD(model,args,VectorData)
    elif args.mode == 'RNN':
        model = RTSRVFD(inc=2,outc=1,init_channels=args.init_channels,num=args.interval)
        if args.cuda:
            model.cuda()
        model.apply(weights_init_kaiming)
        trainTSRVFD(model,args,VectorData)
    elif args.mode == 'GAN':
        model = TSRVFD(inc=2,outc=args.interval,init_channels=args.init_channels,num=1)
        D = Dis()
        if args.cuda:
            model.cuda()
            D.cuda()
        model.apply(weights_init_kaiming)
        D.apply(weights_init_kaiming)
        trainGAN(model,D,args,VectorData)  
    
    

def GetResult():
    '''
    if args.mode == 'TSR':
        model = torch.load(args.model_path+args.dataset+'/TSR-wo-sep-'+str(args.epochs)+'.pth',map_location=lambda storage, loc:storage)
    elif args.mode == 'RNN':
        model = torch.load(args.model_path+args.dataset+'/RNN-'+str(args.epochs)+'.pth',map_location=lambda storage, loc:storage)
    elif args.mode == 'Interpolation':
        model = torch.load(args.model_path+args.dataset+'/Interpolation-'+str(args.epochs)+'.pth',map_location=lambda storage, loc:storage)
    elif args.mode == 'GAN':
        model = torch.load(args.model_path+args.dataset+'/GAN-'+str(args.epochs)+'.pth',map_location=lambda storage, loc:storage)
    model.cuda()
    '''
    VectorData = DataSet(args)
    VectorData.ReadData()
    #inference(model,VectorData,args)
    GetMetrics(VectorData,args)

if __name__== "__main__":
    if args.train=='train':
        main()
    else:
        GetResult()