import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from tqdm import tqdm
import numpy
import os
from densenet import DenseNet3
import conf.config as conf
from data.dataLoader import *
import utilsloss


torch.set_printoptions(precision=10, threshold=100000)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

model_path = ''
cfg = conf.CIFAR10
test_data = CIFAR10_test_data
OOD_data = TinyImageNet_Resize_Test
utilsloss.PEDCC_PATH = cfg['PEDCC_Type']  # 修改使用的PEDCC文OOD_test.py:24件



class Hook():
    def __init__(self,module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = 0
    def hook_fn(self,module,input,output):
        nch = input[0].shape[1]
        mean = input[0].mean([2,3])
        var = input[0].var([2,3])
        netmean = module.running_mean.data
        netvar = module.running_var.data

        r_feature = ((mean-netmean).pow(2))/(0.5*(var+netvar))
        self.r_feature = r_feature

def test():

    net = DenseNet3(depth=100, num_classes=cfg['num_classes'], feature_size=cfg['feature_size'])

    test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net = net.cuda()
    net = net.module
    net = net.eval()
    # print(net)
    dis =[]
    for module in net.modules():
        if isinstance(module,nn.BatchNorm2d):
            dis.append(Hook(module))
    dis.pop()

    w = net.fc.weight.data


    rank = torch.linalg.matrix_rank(w)

    print(rank)


    w = w.t()
    wt = w.t()
    print(w.shape)

    P = torch.mm(wt,w)

    P = torch.linalg.inv(P)

    P = torch.mm(w,P)

    P = torch.mm(P,wt)

    print(net)
    for im, label in tqdm(test_loader):
        if torch.cuda.is_available():
            im, label = im.cuda(), label.cuda()
        out, outf,outfnorm,f,av,bv = net(im)


        distance = torch.zeros(im.shape[0],1).cuda()
        for (idx, mod) in enumerate(dis):
            # if idx <= 60:
            distance = distance + torch.sum(mod.r_feature,dim=1,keepdim=True)

        av = torch.norm(av,dim=1)
        av = torch.norm(av,dim=2)
        av = torch.norm(av,dim=1)


        bv = torch.norm(bv,dim=1)
        bv = torch.norm(bv,dim=2)
        bv = torch.norm(bv,dim=1)

        outlen = torch.norm(outf,dim=1)
        lenf = torch.norm(f,dim=1)
        r5a = bv/av

        score,_ = out.max(1)

        f = f.t()

        p = torch.mm(P,f)

        p = p.t()
        f = f.t()
        plen = p.pow(2).sum(1)
        plen = plen.sqrt()

        flen = f.pow(2).sum(1)
        flen = flen.sqrt()

        gamma = plen/flen

        a2 = out.pow(2).sum(1)
        # a2 = a2*0.9
        a2 = a2*0.9
        a = a2.sqrt()



        for elem in a:
            f1.write("{}\n".format(elem))
        for i in range(score.shape[0]):
            f2.write("{}\n".format((score[i]/a[i])))
        for elem in outlen:
            f4.write("{}\n".format(elem))
        for elem in r5a:
            f3.write("{}\n".format(elem))
        for i in range(gamma.shape[0]):
            f5.write("{}\n".format(gamma[i]))
        for elem in distance:
            f6.write("{}\n".format(elem[0]))

if __name__ == '__main__':
    test()
