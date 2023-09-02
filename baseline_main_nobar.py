'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import random
import os
import argparse
#from data_loader import data_loader
from torch.utils.data import DataLoader
from models import *
#from utils import progress_bar
from subfunction import CIFAR20
from roc_tpr import cal_roc_tpr
parser = argparse.ArgumentParser(description='CIFAR10_Anomaly_Detection')
parser.add_argument('--seed', '-seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', '-gpu', default="0", type=str, help='gpu_id')
parser.add_argument('--beta', '-beta', default=-2.0, type=float, help='penalty_for the lower effective_rank')
parser.add_argument('--eigen', '-eigen', default=28, type=int, help='number of zeroing eigenvalues for the adversary')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark=False
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


 
def _init_fn(worker_id):
    np.random.seed(args.seed)

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# test_32x32.mat
svhn = torchvision.datasets.SVHN(
    root='./data',
    split='test', 
    download=True, 
    transform = transform_test
    )
svhn_loader = torch.utils.data.DataLoader(
    dataset= svhn, 
    batch_size=100, 
    shuffle=False, 
    num_workers=1,
    worker_init_fn= _init_fn
    )

# cifar-10-python.tar.gz  cifar-10-batches-py
# train with the blurred training data
trainset = CIFAR20(
    root='./data',
    train=True, 
    download=True, 
    transform=transform_test,
    reduce_eigenvalue=args.eigen
    )
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=2, 
    worker_init_fn = _init_fn
    )
# cifar-10-python.tar.gz  cifar-10-batches-py
# test with the unblurred test data
testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
    )
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=1,
    worker_init_fn = _init_fn
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

lsun = datasets.ImageFolder(
    "./data/LSUN_resize/",
    transform=transform_test
    )
lsun_loader = DataLoader(
    dataset = lsun, 
    batch_size= 100, 
    shuffle=False, 
    worker_init_fn = _init_fn
    )

timagenet = datasets.ImageFolder(
    "./data/Imagenet_resize/"
    ,transform=transform_test
    )
timagenet_loader = DataLoader(
    dataset = timagenet, 
    batch_size=100, 
    shuffle=False, 
    worker_init_fn = _init_fn
    )  
# Model
print('==> Building model..')
net = ResNet345()
net2= ResNet342()
net3 = ResNet342()
net = net.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net2 = torch.nn.DataParallel(net2)
    net3 = torch.nn.DataParallel(net3)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train() # f
    net2.eval() # g2
    net3.eval() # g3
    # net2 and net3 have the same eigenvalue
    train_loss = 0
    lr_epoch = 1e-4
  
    if epoch>=25:
        lr_epoch *=1e-1
    optimizer = optim.Adam(net.parameters(), lr=lr_epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs.shape = [128, 3, 32, 32]
        # targets.shape = [128]
        optimizer.zero_grad()
        batch_size= targets.size(0)
        inputs = Variable(inputs, requires_grad=True) # why requires_grad = True?
        outputs = net(inputs)     # f(x')  shape = [batch_size(128), 512]
        outputs3,_ = net2(inputs) # g2(x') shape = [batch_size(128), 512]
        outputs5,_ = net3(inputs) # g3(x') shape = [batch_size(128), 512]
        print(outputs3 - outputs5)
        outputs3 = outputs3.data.cpu().numpy()
        outputs5 = outputs5.data.cpu().numpy()
        outputs4 = np.reshape(np.vstack([outputs3,outputs5]), [2,batch_size,512])
        # outputs4.shape = (2, 128, 512)
        outputs6 = np.zeros([batch_size,512]) # np.shape = [128, 512]
        for i in range(batch_size):
            # cannot run, index is float
            # what is this line trying to do?
            outputs6[i,:] = outputs4[targets[i].data.cpu().numpy(),i,:]
        outputs6 = torch.cuda.FloatTensor(outputs6)
        loss1 = torch.sum(torch.abs(outputs-outputs6) ** 2 , 1)
        loss = torch.mean(loss1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))
     
        
def test(epoch):
    net.eval()
    post_element = np.zeros([10 ** 4])
    check_indices= 0
    print('test')
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = Variable(inputs, requires_grad=True)
            outputs = net(inputs)
            outputs3, _ = net2(inputs)
            for i in range(100):
                 sub1 = torch.sum((outputs[i]-outputs3[i])**2)
                 post_element[check_indices]= sub1
                 check_indices += 1
              
    print('test_end')
    return post_element,check_indices
def test2(epoch):
    project_array = np.zeros([26032])
    project_array2 = np.zeros([10 ** 4])
    project_array3 = np.zeros([10 ** 4])
    check_indices = 0
    check_indices2 = 0 
    check_indices3 = 0
    net.eval()
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(svhn_loader):
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs3,_ = net2(inputs)
            for i in range(targets.size(0)):
                sub1 = torch.sum((outputs[i]-outputs3[i])**2)
                project_array[check_indices]= sub1
                check_indices +=1
        for batch_idx,inputs in enumerate(lsun_loader):
            targets=inputs[1]
            inputs = inputs[0].to(device)
            outputs = net(inputs)
            outputs3, _ = net2(inputs)
            for i in range(targets.size(0)):
                sub1 = torch.sum((outputs[i]-outputs3[i])** 2)
                project_array2[check_indices2]= sub1
                check_indices2 +=1            
        for targets,inputs in enumerate(timagenet_loader):
            inputs = inputs[0].to(device)
            targets = np.zeros([100])
            outputs = net(inputs)
            outputs3, _ = net2(inputs)   
            for i in range(100):     
                sub1 = torch.sum((outputs[i]-outputs3[i])**2)
                project_array3[check_indices3]= sub1
                check_indices3 +=1     
    return project_array, check_indices,project_array2, check_indices2, project_array3, check_indices3   
def check_anomaly(post_statistics,project_array,check_indices,project_array2,check_indices2, project_array3, check_indices3):
    #################### First, generate validataion statistics
    val_svhn = project_array[0:1000]
    val_lsun = project_array2[0:1000]
    val_timg = project_array3[0:1000]
    test_svhn = project_array[1000:]
    test_lsun = project_array2[1000:]
    test_timg = project_array3[1000:]
    ##############################################################
    roc_val_svhn, tnr_val_svhn,_,_,_ = cal_roc_tpr(post_statistics,val_svhn,0.95)
    roc_val_lsun, tnr_val_lsun,_,_,_ = cal_roc_tpr(post_statistics,val_lsun,0.95)
    roc_val_timg, tnr_val_timg,_,_,_ = cal_roc_tpr(post_statistics,val_timg,0.95)
    ################################################################
    roc_test_svhn, tnr_test_svhn,svhn_det,svhn_in,svhn_out = cal_roc_tpr(post_statistics,test_svhn,0.95)
    roc_test_lsun, tnr_test_lsun,lsun_det,lsun_in,lsun_out = cal_roc_tpr(post_statistics,test_lsun,0.95)
    roc_test_timg, tnr_test_timg,timg_det,timg_in,timg_out = cal_roc_tpr(post_statistics,test_timg,0.95)
   
    print('svhn data')
    print(tnr_val_svhn)
    print(tnr_test_svhn)
    print('lsun data')
    print(tnr_val_lsun)
    print(tnr_test_lsun)
    print('timg data')
    print(tnr_val_timg)
    print(tnr_test_timg)
    
    return tnr_val_svhn,tnr_val_lsun,tnr_val_timg,roc_test_svhn,tnr_test_svhn,svhn_det,svhn_in,svhn_out,roc_test_lsun,tnr_test_lsun,lsun_det,lsun_in,lsun_out,roc_test_timg,tnr_test_timg,timg_det,timg_in,timg_out


##################################
performancesave = np.zeros([50,18])
abcd = np.zeros([10])
for epoch in range(50):
    train(epoch)
    if epoch% 10 == 9:
       post_statistics, check_index  = test(epoch)
       project_array, test_length,project_array2,test_length2,project_array3,test_length3 = test2(epoch)
       print(test_length)
       print(test_length2)
       print(test_length3)
       print(check_index)
       performancesave[epoch,:] = check_anomaly(post_statistics,project_array,test_length,project_array2,test_length2, project_array3,test_length3)
#       abcd = top10(post_statistics)
epsilon_start = str(args.eigen)+'_'
epsilon_num = 'epsilon%d.txt' %args.seed
epsilon_num = epsilon_start+epsilon_num
np.savetxt(epsilon_num,performancesave)
