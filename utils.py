 # -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018
 
@author: gk
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, Conv3d, ModuleList, Parameter, LayerNorm, BatchNorm1d, BatchNorm3d


"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""
    

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(multi_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
    
class NTXentLoss1(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLoss1, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.bn=BatchNorm1d(batch_size,affine=False)
    def forward(self, zis, zjs):
        shape=zis.shape
        zis=zis.reshape(shape[0],-1)
        zjs=zjs.reshape(shape[0],-1)
        
        zis1 = F.normalize(zis, p=2, dim=-1)
        zjs1 = F.normalize(zjs, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(zis1,zjs1.permute(1,0))
        similarity_matrix1 = torch.matmul(zis1,zis1.permute(1,0))
        shape=similarity_matrix.shape
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix)
        positives = l_pos#.view(shape[0],self.batch_size, 1)
        positives = positives/self.temperature
        
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask).type(torch.bool).cuda()
        negatives = similarity_matrix[mask].view(self.batch_size,self.batch_size-1)
        negatives =negatives /self.temperature

        
        loss=-positives+torch.logsumexp(negatives, dim=-1, keepdim=False)
        # loss = -torch.log((torch.exp(positives))/(torch.exp(positives)\
        #                                                           +torch.sum(torch.exp(negatives),-1,True)))
        return loss.mean()
    
class ST_BLOCK_8(nn.Module):
    def __init__(self,device,c_in,c_out,num_nodes,dropout,tem_size,K,Kt):
        super(ST_BLOCK_8,self).__init__()
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        self.multigcn=multi_gcn(c_out,2*c_out,dropout,support_len=3)
        
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.dropout=dropout
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        
        self.c_out=c_out
        self.bn=BatchNorm2d(2*c_out)
        
        self.conv1=Conv2d(c_in, 2*c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
    def forward(self,x,support,supports):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        
        x_s1=self.gcn(x,support)   
        x1,x2=torch.split(x_s1,[self.c_out,self.c_out],1)
        x_s1=F.relu(x1)*(torch.sigmoid(x2))
        
        x_s2=self.multigcn(x,supports)   
        x3,x4=torch.split(x_s2,[self.c_out,self.c_out],1)
        x_s2=F.relu(x3)*(torch.sigmoid(x4))
        x=torch.cat((x_s1,x_s2),1)
        out=self.bn(x+residual[:,:,:,-x.size(3):])
        return out
    

