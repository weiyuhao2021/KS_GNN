import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, ChebConv  



class MLP(torch.nn.Module):
    def __init__(self, input_dim,  hidden_dim, output_dim, feature_dim = None,
                 feature_pre=False, layer_num=2, dropout=False, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        # x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        return x



class KSNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, l1, l2, l3, l4, layer_num=2, conv_num = 5, alpha=0.9):
        # super(KSNN, self).__init__(aggr='max')
        super(KSNN, self).__init__(aggr='mean')
        
        self.input_dim = input_dim
        self.encoder = MLP(self.input_dim, hidden_dim, output_dim, layer_num = layer_num)
        self.decoder = MLP(output_dim, hidden_dim, self.input_dim, layer_num = layer_num)     
        self.alpha = alpha
        self.conv_num = conv_num
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.pca_Z = None
        self.pca_v = None

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        
    def msg(self, x, edge_index):
        # breakpoint()
        de_X = self.decoder(x)
        x_j = self.propagate(edge_index, x=de_X)*self.alpha
        # tmp_x = torch.max(de_X,x_j)
        # tmp_x = (de_X+x_j)/2

        # device = torch.device(f"cuda:{0}")
        # Nv = torch.zeros(3327)
        # Nv = Nv.to(device)
        # for i in edge_index[0]:
        #     Nv[i]=Nv[i]+1
        #
        # # breakpoint()
        # Nv_ = torch.Tensor(3703,3327)
        # Nv_ = Nv_.to(device)
        # Nv_ = Nv_.copy_(Nv)
        # Nv_ = Nv_.transpose(0,1)
        # one = torch.ones(3327, 3703)
        # one = one.to(device)
        # l1 = torch.div(Nv_, Nv_ + one)
        # l2 = torch.div(one, Nv_ + one)
        # breakpoint()
        tmp_x = torch.mul(self.l1, x_j)+torch.mul(self.l2, de_X)

        # Nv_ = torch.Tensor(3327, 3327)
        # Nv_ = Nv_.copy_(Nv)
        # Nv_ = Nv_.to(device)
        # one=torch.ones(3327, 3327)
        # one = one.to(device)
        # l1 = torch.div(Nv_,Nv_ + one)
        # l2 = torch.div(one, Nv_ + one)
        # tmp_x = torch.matmul(x_j.transpose(0,1),l1) + torch.matmul(de_X.transpose(0,1),l2)
        # tmp_x = tmp_x.transpose(0,1)

        # tmp_x_j = torch.Tensor(3327,3703)
        # i = 0
        # for j in x_j:
        #     # tmp_x_j[i] = j*Nv[i]/(Nv[i]+1)
        #     # tmp_x = tmp_x_j + de_X / (Nv[i] + 1)
        #     i=i+1
        # breakpoint()
        return self.encoder(tmp_x)

    def kspca_msg(self, x, edge_index):
        de_X = x @ self.pca_v.t()
        x_j = self.propagate(edge_index, x=de_X)*self.alpha
        # tmp_x = torch.max(de_X,x_j)
        # tmp_x = (de_X + x_j) / 2
        tmp_x = torch.mul(self.l1,x_j)+torch.mul(self.l2,de_X)
        return tmp_x @ self.pca_v

    def kspca(self, X, edge_index):
        # breakpoint()
        x = self.pca_Z if not self.pca_Z is None else self._set_pcaX(X)
        for i in range(self.conv_num):
            x = self.kspca_msg(x,edge_index)
        return x

    def forward(self, x, edge_index):
        # breakpoint()
        x = self.encoder(x)
        for i in range(self.conv_num):
            x = self.msg(x, edge_index)
        return x

    def sage_msg(self, x, edge_index):
        de_X = x
        x_j = self.propagate(edge_index, x=de_X)
        # return torch.max(de_X,x_j)
        # return (de_X+x_j)/2
        tmp_x = torch.mul(self.l3, x_j) + torch.mul(self.l4, de_X)
        return tmp_x

    def sage_forward(self, x, edge_index):
        x = self.encoder(x)
        for i in range(self.conv_num):
            x = self.sage_msg(x, edge_index)
        return x

    def _set_pcaX(self, X):
        u,s,v = torch.svd(X)
        self.pca_v = v[:,:self.output_dim]
        self.pca_Z = X @ self.pca_v
        return self.pca_Z

    def pca_msg(self, x, edge_index):
        x_j = self.propagate(edge_index, x=x)*self.alpha
        # return torch.max(x,x_j)
        # return (x+x_j)/2
        tmp_x = torch.mul(self.l3, x_j) + torch.mul(self.l4, x)
        return tmp_x

    def pca(self, X, edge_index):
        x = self.pca_Z if not self.pca_Z is None else self._set_pcaX(X)
        for i in range(self.conv_num):
            x = self.pca_msg(x,edge_index)
        return x
    

