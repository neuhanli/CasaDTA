import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict

import torch

import torchvision.transforms as transforms
from PIL import Image
import networkx as nx
from torch.nn.utils.rnn import pad_sequence

from utils import *


def prot_cat(prot):
    x = np.zeros(max_prot_len)
    for i, ch in enumerate(prot[:max_prot_len]):
        x[i] = prot_dict[ch]
    return x

def drug_cat(drug):
    x = np.zeros(max_drug_len)
    for i, ch in enumerate(drug[:max_drug_len]):
        x[i] = drug_dict[ch]
    return x

#蛋白质
prot_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
prot_dict = {v:(i+1) for i,v in enumerate(prot_voc)}
prot_dict_len = len(prot_dict)
max_prot_len = 1000

#药物
drug_voc = "#%)(+-/.1032547698=A@CBEDGFIHKMLONPSRUTWVY[Z]\\acbedgfihmlonsruty"
drug_dict = {v:(i+1) for i,v in enumerate(drug_voc)}
drug_dict_len = len(drug_dict)
max_drug_len = 100


class Data:
    def __init__(self, x, y, target_Embedding):
        self.x = x
        self.y = y
        self.target_Embedding = target_Embedding


def collate_fn(data):
    x = [item.x for item in data]
    x=torch.cat(x)
    y = [item.y for item in data]
    y=torch.cat(y)

    target_Embedding = [item.target_Embedding for item in data]
    target_Embedding = torch.cat(target_Embedding, 0)


    return x, y, target_Embedding


def process(XD,XT_Embedding,XY):
    assert len(XD) == len(XY) == len(XT_Embedding), "The three lists must be the same length!"
    data_list = []
    data_len = len(XD)
    for i in range(data_len):
        print('Converting Data: {}/{}'.format(i+1, data_len))
        smiles = XD[i]
        target_Embedding = XT_Embedding[i]

        labels = XY[i]
        x = torch.LongTensor([smiles])
        y = torch.FloatTensor([labels])
        target_Embedding = torch.LongTensor([target_Embedding])

        data = Data(x=x, y=y, target_Embedding=target_Embedding)
        data_list.append(data)
    return data_list
def loadTrain(dt_name):


    XD_train=[]
    XT_ESM_train=[]
    XT_Embedding_train=[]
    XY_train=[]
    datasets = ['davis']
    # convert to PyTorch data format
    for dataset in datasets:
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots, train_Y, dip = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity']), list(df['dip'])

        XT_Embedding = [prot_cat(t) for t in train_prots]

        XD = [drug_cat(t) for t in train_drugs]


        XD_train, XT_Embedding_train, XY_train = np.asarray(XD), np.asarray(XT_Embedding), np.asarray(train_Y)
    train_data = process(XD_train,XT_Embedding_train,XY_train)
    return train_data
def loadTest(dt_name):
    compound_iso_smiles = []
    df = pd.read_csv('data/' + dt_name + '_' + 'test' + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)

    XD_test=[]
    XT_Embedding_test=[]
    XT_ESM_test=[]
    XY_test=[]
    datasets = ['davis']
    # convert to PyTorch data format
    for dataset in datasets:
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_Y, dip= list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity']), list(df['dip'])

        XT_Embedding = [prot_cat(t) for t in test_prots]

        XD = [drug_cat(t) for t in test_drugs]
        XD_test, XT_Embedding_test, XY_test = np.asarray(XD), np.asarray(XT_Embedding), np.asarray(test_Y)
    test_data = process(XD_test, XT_Embedding_test, XY_test)
    return test_data


