import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import torch.autograd as autograd
 
import numpy as np
import math
import random
from torch.nn import BatchNorm2d, BatchNorm1d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d, Dropout2d
import util

from utils import ST_BLOCK_8 

from utils import NTXentLoss1

def TS(x):
    shape=x.shape
    random=np.random.uniform(0,1,size=shape)
    random1= torch.from_numpy(random).float().cuda()
    x_pool=F.avg_pool2d(x,kernel_size=[1,3],stride=[1,1],padding=[0,1])
    x=x_pool*(random1)+x*(1-random1)
    
    return x

def mask(x,alpha):
    random=np.random.uniform(0,1,size=x.shape)
    random1= torch.from_numpy(random).float().cuda()
    mask=(random1>alpha).float()
    x=mask*x
    return x

def random_transformer(x,alpha):
    x=TS(x)
    x=mask(x,alpha)
    return x

class base_model(nn.Module):
    def __init__(self,device,num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2):
        super(base_model, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.c_out=dilation_channels
        self.supports = supports
        tem_size=length
        K=3
        self.start_conv1=nn.Conv2d(in_channels=in_dim,
                                  out_channels=dilation_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.block1=ST_BLOCK_8(device,in_dim,dilation_channels,num_nodes,dropout,tem_size,K,5)
        self.block2=ST_BLOCK_8(device,2*dilation_channels,dilation_channels,num_nodes,dropout,tem_size-4,K,5)
        self.block3=ST_BLOCK_8(device,2*dilation_channels,dilation_channels,num_nodes,dropout,tem_size-8,K,4)
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
    def forward(self,input):
        
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,self.dropout,self.training)
        
        
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        
        loss1=torch.zeros(1).cuda()
        skip=0    
        #x_input=self.start_conv1(input)
        x1=self.block1(input,A,new_supports) 
        x2=self.block2(x1,A,new_supports)
        x3=self.block3(x2,A,new_supports)    
        
        return x1,x2,x3
        

class CLDGCN(nn.Module):
    def __init__(self,device,num_nodes, dropout=0.3, supports=None, CL=False, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2):
        super(CLDGCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.c_out=dilation_channels
        self.supports = supports
        self.CL=CL
        tem_size=length
        K=3
        
        
        self.model=base_model(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=out_dim, 
                           residual_channels=residual_channels, dilation_channels=dilation_channels, 
                           skip_channels=skip_channels, end_channels=end_channels)
        
        self.projection_head=nn.Sequential(
            #nn.BatchNorm2d(skip_channels),
            Conv2d(2*dilation_channels,2*dilation_channels//4,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False),
            #nn.BatchNorm2d(2*dilation_channels//4),
            nn.ReLU(inplace=True),
            Conv2d(2*dilation_channels//4,2*dilation_channels,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False) 
            )
        
        
        self.loss=NTXentLoss1(device,64,0.1)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv3=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.gate = nn.Conv2d(in_channels=2*skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=(1,1),
                                  bias=False)
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
    def forward(self,input):
        
        shape=input.shape
        loss1=torch.zeros(1).cuda()
        
        if self.CL:
            input1=random_transformer(input,0.1) #attention, according datasets select value->04: 0.3；08: 0; 03: 0; 07: 0.1
            x_1,x_2,x_3=self.model((input1))
            
            input2=random_transformer(input,0.1) #attention, according datasets select value->04: 0.3；08: 0; 03: 0; 07: 0.1
            x_4,x_5,x_6=self.model(((input2)))
            
            p=(self.projection_head(x_3))
            p1=(self.projection_head(x_6))
            
            loss1= self.loss(p,p1)
            x=torch.zeros(shape[0],12,shape[2],1).cuda()
            
        else:
            skip=0
            
            x1,x2,x3=self.model(input)
            skip=skip+self.skip_conv1(x1)
            skip=skip[:,:,:,-x2.size(3):]+self.skip_conv2(x2)   
            skip=skip[:,:,:,-x3.size(3):]+self.skip_conv3(x3)
            skip=F.relu(skip)
            x = F.relu(self.end_conv_1(skip))
            x = self.end_conv_2(x)
        return x,(loss1),loss1


