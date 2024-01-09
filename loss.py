import torch
import torch.nn.functional as F
import numpy as np
import math
import scipy.stats as st

'''
Reference:
@inproceedings
{
    yuan2022compositional,
    title = {Compositional Training for End - to - End Deep AUC Maximization},
    author={Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net / forum?id = gPvB4pdu_Z}
}

@article{
    Xu2024FAUC-S,
    title={FAUC-S:Deep AUC maximization by focusing on hard samples},
    author={Shoukun Xu, Yanrui Ding, Yanhao Wang, Junru Luo},
    journal={Neurocomputing},
    year={2024},
}
'''

class focal_AUC_loss(torch.nn.Module):
    """
        FAUC-S Loss: a novel loss function to directly optimize AUROC
        inputs:

            margin: margin term for AUCM loss, e.g., m in [0, 1]
            imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
        outputs:
            loss
    """
    def __init__(self, imratio=None,  margin=1, backend='ce',device=None,gama=1,j=1/2):
        super(focal_AUC_loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.gama = gama
        self.j = j
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.L_AVG = F.binary_cross_entropy_with_logits  # with sigmoid

        self.backend = 'ce'  #TODO:


    def forward(self, y_pred, y_true):
        if len(y_pred) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true) == 1:
            y_true = y_true.reshape(-1, 1)
        if self.backend == 'ce':
           self.backend = 'auc'
           return self.L_AVG(y_pred,y_true)
        else:
           self.backend = 'ce'
           if self.p is None:
              self.p = (y_true==1).float().sum()/y_true.shape[0]
           y_pred = torch.sigmoid(y_pred)

           # ##exponential function
           # p1 = torch.exp(-self.gama*(0.5*(y_pred-self.a)))
           # p2 = torch.exp(-self.gama*(0.5*(self.b-y_pred)))

           ##logarithmic function
           # p1 = torch.log2(1+torch.exp(-self.gama*(0.5*(y_pred-self.a))))
           # p2 = torch.log2(1+torch.exp(-self.gama*(0.5*(self.b-y_pred))))

           ##power function
           p1 = (1-self.j*(y_pred-self.a)) ** self.gama
           p2 = (1-self.j*(self.b-y_pred)) ** self.gama

           ##ours
           p11 = p1*(1 == y_true).float()
           p22 = p2*(0 == y_true).float()
           pp = torch.mean(p11+p22)
           self.L_AUC = (1 - self.p) * torch.mean(p1 * (y_pred - self.a) ** 2 * (1 == y_true).float()) + \
                        (self.p) * torch.mean(p2 * (y_pred - self.b) ** 2 * (0 == y_true).float()) + \
                        2 * self.alpha * (self.p * (1 - self.p) * self.margin*pp+ \
                                          torch.mean((self.p * p2 * y_pred * (0 == y_true).float() - \
                                                      (1 - self.p) * p1 * y_pred * (1 == y_true).float()))) - \
                        self.p * (1 - self.p) * self.alpha ** 2*pp
           return self.L_AUC
