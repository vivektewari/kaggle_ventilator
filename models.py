import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math
class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy

class fc_model(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, bias=True):
        super(fc_model, self).__init__()
        self.input_size = input_size
        self.hidden_size=hidden_size
        self.bias = bias
        self.activation=torch.nn.LeakyReLU()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2= nn.Linear(hidden_size, int(hidden_size/2), bias=bias)
        self.fc3 = nn.Linear(int(hidden_size/2), output_size, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)




class GRUModel(nn.Module):# for 1 rc value only and different fc for each seq
    def __init__(self, input_dim, hidden_dim, output_dim,seq_count, rc_seq=0):#ori: layer_dim,
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        #ori self.layer_dim = layer_dim

        self.gru_cells = nn.ModuleList()
        self.fc1s=nn.ModuleList()
        #self.fc2 = nn.ModuleList()
        for i in range(9):
            self.fc1s.append(nn.Linear(hidden_dim, output_dim) )
            self.gru_cells.append(GRUCell(input_dim, hidden_dim))
            #self.fc2.append(nn.Linear(hidden_dim, output_dim))
        self.rc_seq=rc_seq
        self.update_rc_seq()
        self.activation_l = torch.nn.LeakyReLU()

    def clear_rc_seq(self):
        self.gru_cell = None
        self.fc1 = None

    def update_rc_seq(self):

        self.gru_cell=self.gru_cells[self.rc_seq]
        self.fc1=self.fc1s[self.rc_seq]
    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28

        x=x[1]
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))

        outs,out = [],[]
        out_val=torch.zeros((x.shape[0],40))
        hn = h0[0, :, :]
        #out=torch.zeros((x.shape))

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
            out = self.fc1(hn)
            out = self.activation_l(out)
            # out = self.fc2[seq](out)
            # out = self.activation_l(out)
            out_val[:,seq]=out.squeeze()


        return out_val

class fc_model_holder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,seq_count, bias=True):#ori: layer_dim,
        super(fc_model_holder, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        self.fc_model_list = nn.ModuleList()


        for i in range(seq_count):
            self.fc_model_list.append(fc_model(input_dim*(i+1),hidden_dim*(i+1),output_dim))
            #self.fc2.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, x):


        x=x[1]


        out_val=torch.zeros((x.shape[0],40))

        #x=torch.unsqueeze(x,2)
        for seq in range(x.size(1)):

            out =self.fc_model_list[seq](x[:, :seq+1])
            out_val[:,seq]=out.squeeze()


        return out_val