import torch.nn as nn
import torch
class p1_Net(nn.Module):
    def __init__(self):
        super(p1_Net,self).__init__()
        self.l1 = nn.Linear(in_features=6,out_features=50,bias=True,dtype=torch.float64)
        self.relu1 = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu2 = nn.ReLU(inplace=False)
        self.l3 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu3 = nn.ReLU(inplace=False)
        self.l4 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu4 = nn.ReLU(inplace=False)
        self.l5 = nn.Linear(in_features=50,out_features=50,dtype=torch.float64)
        self.relu5 = nn.ReLU(inplace=False)
        self.l6 = nn.Linear(in_features=50, out_features=2, dtype=torch.float64)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        out = self.relu5(out)
        out = self.l6(out)
        return out
class p2_Net(nn.Module):
    def __init__(self):
        super(p2_Net,self).__init__()
        self.l1 = nn.Linear(in_features=6,out_features=50,bias=True,dtype=torch.float64)
        self.relu1 = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu2 = nn.ReLU(inplace=False)
        self.l3 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu3 = nn.ReLU(inplace=False)
        self.l4 = nn.Linear(in_features=50,out_features=50,bias=True,dtype=torch.float64)
        self.relu4 = nn.ReLU(inplace=False)
        self.l5 = nn.Linear(in_features=50,out_features=50,dtype=torch.float64)
        self.relu5 = nn.ReLU(inplace=False)
        self.l6 = nn.Linear(in_features=50, out_features=4, dtype=torch.float64)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        out = self.relu5(out)
        out = self.l6(out)
        return out
