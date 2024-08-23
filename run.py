import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from step_learning import *
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='Few-Shot GAD')
parser.add_argument('--dataset', type=str, choices=['injected_cora', 'injected_citeseer', 'injected_amazon_photo', 'wiki', 'amazon_review','yelpchi'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device')
parser.add_argument('--encoder_lr', type=float, default=5e-3)
parser.add_argument('--adaptor_lr', type=float)
parser.add_argument('--adaptor_weight_decay', type=float, default=0)
parser.add_argument('--detector_lr', type=float)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--grad_clip', type=float, default=5)
parser.add_argument('--train_per', type=float, default=0.8)
parser.add_argument('--val_per', type=float, default=0.9)
parser.add_argument('--lambda_our', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--few_shot', type=int, default=10)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--num_run', type=int, default=1)
parser.add_argument('--pos_weight', type=float, default=1)
parser.add_argument('--readout', type=str, default='avg')  #avg max min weighted_sum
parser.add_argument('--contamination', action='store_true')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Dataset: ',args.dataset)

# Load and preprocess data
adj, feature, label, embedding = load_data(args)
idx_train_ano_all, idx_train_normal_all, idx_val, idx_test = split_data(label, args)
num_nodes, ft_dim = feature.shape

# MetaGAD
results_roc = []
results_pr = []
for t in range(args.num_run):
    print('%dth run, training %d epochs, '%(t, args.num_epoch), end='')

    labeled_idx, unlabled_idx = sample_anomaly(idx_train_ano_all, idx_train_normal_all, args.few_shot)
    
    best_loss = 1e9
    best_training_epoch = 0

    epoch_list = []
    loss_train_list = []
    loss_val_list = []

    base_model = Detector_mlp(args.embedding_dim).to(args.device)
    base_opt = torch.optim.SGD(base_model.parameters(), lr = args.detector_lr, momentum = args.momentum, weight_decay = args.weight_decay)

    meta_model = Adaptor(args.embedding_dim).to(args.device)
    meta_opt = torch.optim.Adam(meta_model.parameters(), lr = args.adaptor_lr, betas=(0.9, 0.999), weight_decay=args.adaptor_weight_decay)

    for epoch in range(args.num_epoch):
        base_model.train()
        meta_model.train()

        support_idx, label_task = task_generator_full_unlabeled(labeled_idx, unlabled_idx)

        label_task = torch.unsqueeze(torch.tensor(label_task, dtype=torch.float32), 1).to(args.device)
        label_val = torch.unsqueeze(torch.tensor([label[idx] for idx in idx_val], dtype=torch.float32), 1).to(args.device)
        label_test = torch.unsqueeze(torch.tensor([label[idx] for idx in idx_test], dtype=torch.float32), 1).to(args.device)
        
        loss_train, loss_val = step_metalearning_mlp(base_model, base_opt, meta_model, meta_opt, args.detector_lr, 
                                                        embedding, support_idx, label_task, idx_val, label_val, args)

        loss_train = loss_train.detach().cpu().numpy()
        loss_val = loss_val.detach().cpu().numpy()

        epoch_list.append(epoch)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_training_epoch = epoch
            torch.save(base_model.state_dict(), 'saved_model/best_base_model_' + args.dataset + '.pkl')
            torch.save(meta_model.state_dict(), 'saved_model/best_meta_model_' + args.dataset + '.pkl')

    print('traning finished, loading %dth epoch, '%(best_training_epoch), end='')
    base_model.load_state_dict(torch.load('saved_model/best_base_model_' + args.dataset + '.pkl'))
    meta_model.load_state_dict(torch.load('saved_model/best_meta_model_' + args.dataset + '.pkl'))

    base_model.eval()
    meta_model.eval()

    cur_roc, cur_pr = model_test_both(base_model, meta_model, embedding, idx_test, label_test)
    print('AUC_ROC: %.4f, AUC_PR: %.4f'%(cur_roc, cur_pr))
    results_roc.append(cur_roc)
    results_pr.append(cur_pr)

print('3 run results, AUC_ROC: %.4f+-%.4f; AUC_PR: %.4f+-%.4f'%(np.average(results_roc[:3]), np.std(results_roc[:3]), 
                                                                    np.average(results_pr[:3]), np.std(results_pr[:3])))
