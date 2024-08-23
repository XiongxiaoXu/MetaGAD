from inspect import trace
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
#import dgl
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk

def model_test_both(base_model, meta_model, embedding, idx_test, label_test):
    input_test = embedding[idx_test]
    with torch.no_grad():
        logits = base_model(meta_model(input_test))
        logits = torch.squeeze(logits)
        logits = torch.sigmoid(logits)

    ano_score = logits.cpu().numpy()
    label_test = label_test.cpu().numpy()

    fpr, tpr, _ = roc_curve(label_test, ano_score)
    cur_roc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label_test, ano_score)
    cur_pr = auc(recall, precision)

    return cur_roc, cur_pr

def task_generator_full_unlabeled(labeled_idx, unlabled_idx):
    labeled_idx = np.random.permutation(labeled_idx)
    unlabled_idx = np.random.permutation(unlabled_idx)

    support_idx = labeled_idx.tolist() + unlabled_idx.tolist()
    label_task = np.concatenate((np.ones(len(labeled_idx)), np.zeros(len(unlabled_idx))))

    return support_idx, label_task

def sample_anomaly(idx_train_ano_all, idx_train_normal_all, few_shot):
    idx_train_ano_all = np.random.permutation(idx_train_ano_all)
    idx_train_normal_all = np.random.permutation(idx_train_normal_all)

    labeled_idx = idx_train_ano_all[:few_shot]
    unlabled_idx = np.random.permutation(np.concatenate((idx_train_normal_all, idx_train_ano_all[few_shot:])))

    return labeled_idx, unlabled_idx

def split_data(label, args):
    idx_anomaly = np.random.permutation(np.squeeze(np.nonzero(label==1)))
    idx_normal = np.random.permutation(np.squeeze(np.nonzero(label==0)))

    split_ano_train = int(args.train_per * len(idx_anomaly))
    split_normal_train = int(args.train_per * len(idx_normal))
    split_ano_val = int(args.val_per * len(idx_anomaly))
    split_normal_val = int(args.val_per * len(idx_normal))

    idx_train_ano_all = idx_anomaly[:split_ano_train]
    idx_train_normal_all = idx_normal[:split_normal_train]
    idx_val = np.concatenate((idx_anomaly[split_ano_train:split_ano_val], idx_normal[split_normal_train:split_normal_val]))
    idx_test = np.concatenate((idx_anomaly[split_ano_val:], idx_normal[split_normal_val:]))

    return idx_train_ano_all, idx_train_normal_all, idx_val, idx_test

def load_data(args):
    embedding = torch.load('./embedding/embedding_{}.pt'.format(args.dataset))

    if args.dataset.startswith('injected'):
        data = torch.load('./data/injected_dataset_pt/{}.pt'.format(args.dataset))
    else:
        data = torch.load('./data/organic_dataset_pt/{}.pt'.format(args.dataset))

    adj = to_dense_adj(data.edge_index)[0]
    feature = data.x
    label = data.y

    args.embedding_dim = embedding.shape[1]

    return adj, feature, label, embedding
