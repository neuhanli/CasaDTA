import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
from transformer_xtd import SMILES_FASTAModel_xtd

import torch.nn.functional as F
class MLP_xd(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP_xd, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.norm = nn.BatchNorm1d(in_channels, eps=1e-05)  # 使用更常见的eps值
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(drop_rate)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
class MLP_xt(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP_xt, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.norm = nn.BatchNorm1d(in_channels, eps=1e-05)  # 使用更常见的eps值
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(drop_rate)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
class ConvolutionalAttention_xt(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        inter_channels (int, optional): The channels of intermediate feature.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_heads=8):
        super(ConvolutionalAttention_xt, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))

    def _act_dn(self, x):
        x_shape = x.shape  # n,len,d

        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,len,d -> n,heads,len//heads,d
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, -1])
        return x

    def forward(self, x):
        """
            x (Tensor): The input tensor. (n,len,dim)
        """
        x = self.norm(x)
        x1 = F.conv1d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=1)
        x1 = self._act_dn(x1)
        x1 = F.conv1d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=1)
        x3 = F.conv1d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=1)
        x3 = self._act_dn(x3)
        x3 = F.conv1d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=1)
        x = x1 + x3
        return x
class CABlock_xt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.2):
        super(CABlock_xt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.attn = ConvolutionalAttention_xt(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp = MLP_xt(self.out_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class ConvolutionalAttention_xd(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        inter_channels (int, optional): The channels of intermediate feature.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_heads=8):
        super(ConvolutionalAttention_xd, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))

    def _act_dn(self, x):
        x_shape = x.shape  # n,len,dim
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,len,d -> n,heads,len//heads,d

        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, -1])
        return x

    def forward(self, x):
        """
            x (Tensor): The input tensor. (n,len,dim)
        """
        x = self.norm(x)
        x1 = F.conv1d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=1)
        x1 = self._act_dn(x1)
        x1 = F.conv1d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=1)
        x3 = F.conv1d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=1)
        x3 = self._act_dn(x3)
        x3 = F.conv1d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=1)
        x = x1 + x3
        return x
class CABlock_xd(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.2):
        super(CABlock_xd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.attn = ConvolutionalAttention_xd(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp = MLP_xd(self.out_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x





class CasaNet(torch.nn.Module):

    def __init__(self, num_features_xd=64, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=32, output_dim=128, dropout=0.2):
        super(CasaNet, self).__init__()

        # # drug
        self.embedding_xd = nn.Embedding(65, 128)
        self.conv1_xd = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=4, stride=1)
        self.conv2_xd = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=8, stride=1)
        self.conv3_xd = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=12, stride=1)

        # protein
        self.embedding_xt = nn.Embedding(26, 128)
        self.conv1_xt = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=8, stride=1)
        self.conv2_xt = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=16, stride=1)
        self.conv3_xt = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=24, stride=1)


        self.transformer_xtd = SMILES_FASTAModel_xtd(65, 26)

        self.line = nn.Linear(1100 * 128, 256)

        #alignment
        self.cf_block_xd= CABlock_xd(in_channels=96, out_channels=96)
        self.cf_block_xt = CABlock_xt(in_channels=96, out_channels=96)


        # combined layers
        self.fc1 = nn.Linear(484, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)


        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, smiles, protein_Embedding):
        #drug
        #CNN
        xd = self.embedding_xd(smiles).transpose(1,2)
        xd = self.conv1_xd(xd)
        xd = self.conv2_xd(xd)
        xd = self.conv3_xd(xd)
        xdd = xd


        xdd = self.cf_block_xd(xdd)
        xd, _ = torch.max(xd, -1)



        #protein
        #CNN
        xt = self.embedding_xt(protein_Embedding).transpose(1,2)
        xt = self.conv1_xt(xt)
        xt = self.conv2_xt(xt)
        xt = self.conv3_xt(xt)

        xtt = xt


        xtt = self.cf_block_xt(xtt)
        xt, _ = torch.max(xt, -1)

        xdd = torch.squeeze(xdd)
        xtt = torch.squeeze(xtt)

        xdd, _ = torch.max(xdd, -1)
        xtt, _ = torch.max(xtt, -1)

        xd_emb = self.embedding_xd(smiles)
        xt_emb = self.embedding_xt(protein_Embedding)

        out,attention = self.transformer_xtd(xd_emb, xt_emb)
        attention = torch.squeeze(attention,1)
        # 将attention每一列最大值的索引拿出来，组成一个向量
        max_indices = torch.argmax(attention, dim=1)
        max_indices = max_indices + 1
        xtd, _ = torch.max(attention, 1)

        out_xtd = torch.cat((xdd, xtt,xtd,xd,xt), 1)

        # add some dense layers
        xc = self.fc1(out_xtd)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out_xtd = self.out(xc)

        # 返回预测结果和对齐损失
        return out_xtd,max_indices






