import torch

class model_postpros(nn.Module):
    def __init__(self,pretrainedmodel,ks,cn):
        super(model_postpros, self).__init__()  # same with  res_fdcs_v5
        self.pretrainedmodel=pretrainedmodel

        self.ks=ks
        self.cn=cn
        self.softmax=torch.softmax
        self.SC=Conv2d_symmetric(self.ks,elf.cn)



    def forward(self, x):
        x_size = x.shape
        output_test=self.pretrainedmodel(x)
        out_first_softmax = self.softmax(output_test, dim=1)
        out=self.SC(out_first_softmax)

        return out