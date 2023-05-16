import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn 
import numpy as np

class Conv2d_symmetric(nn.Module):
    def __init__(self,ks,cn):
        super(Conv2d_symmetric, self).__init__()
        self.ks=ks  # Kernel size defined by user
        self.bias = None
        self.stride = 1
        self.padding =int((self.ks-1)/2)
        self.dilation = 1
        self.groups = cn  # Class number 
        self.kernelsize=self.ks
        self.mid_row=self.kernelsize//2+1
        self.kv=[]
        self.CS=False
        self.maxnum=int((self.kernelsize**2+self.kernelsize%2)/2)
        if self.CS:
            self.kv=nn.ParameterList([nn.Parameter(torch.tensor([1/(self.kernelsize**2)],requires_grad=True)) for i in range(self.maxnum)])

        else:
            self.kv=nn.ParameterList([nn.Parameter(torch.rand(1,requires_grad=True)) for i in range(self.maxnum)])
            

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = torch.device('cuda')

        weight = torch.zeros((self.groups,1,self.kernelsize,self.kernelsize)).to(device)
        n=self.kernelsize
        c=0
        for i in range(self.mid_row):
            for j in range(self.kernelsize):
                if c==len(self.kv):
                    break
                
                weight[:,0,i,j]=self.kv[c][0]
                weight[:,0,n - i - 1,n - j - 1]=self.kv[c][0]
                c+=1
                
#         print("weight= ", weight)
#         print("inout = ", input)
        return F.conv2d(input, weight, self.bias, self.stride,self.padding, self.dilation, self.groups)