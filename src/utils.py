import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
from collections import defaultdict
from typing import Sequence
import torch
from src.calibloss import \
    NodewiseNLL, NodewiseBrier, NodewiseECE, EdgewiseNLL, EdgewiseBrier, \
    EdgewiseECE, ECE, Reliability

edge_choices = (
    'full',  # use all available edges including the training part
    'any',  # use edges where at least one node is in mask
    'both'  # use edges where both nodes are in mask
)

def set_global_seeds(seed):
    """
    Set global seed for reproducibility
    """  
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)

def arg_parse():
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora','Citeseer', 'Pubmed'])
    parser.add_argument('--split_type', type=str, default='5_3f_85', help='k-fold and test split')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GMNN', 'EPFGNN'])
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    parser.add_argument('--eval-edge-type', type=str, default='both', choices=edge_choices,  help='edge choice for edgewise eval metrics')
    parser.add_argument('--ece-bins', type=int, default=15, help='number of bins for ece')
    parser.add_argument('--ece-scheme', type=str, default='equal_width', choices=ECE.binning_schemes, help='binning scheme for ece')
    parser.add_argument('--ece-norm', type=float, default=1.0, help='norm for ece')
    parser.add_argument('--reli_diag', action='store_true', default=False)    
    parser.add_argument('--save_prediction', action='store_true', default=False)

    gmnn_parser = parser.add_argument_group('optional GMNN arguments')
    gmnn_parser.add_argument('--iter', type=int, default=1, help='Number of training iterations.')
    gmnn_parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
    gmnn_parser.add_argument('--tau', type=float, default=1, help='Annealing temperature in sampling.')
    gmnn_parser.add_argument('--draw', type=str, default='smp', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
    gmnn_parser.add_argument('--inference', type=str, default='p', help='Which model to use at inference. Do not need to specify in training stage.')
    gmnn_parser.add_argument('--num_samples', type=int, default=32, help='Number of samples for p model inference.')

    epf_parser = parser.add_argument_group('optional EPFGNN arguments')
    epf_parser.add_argument('--epf_iter', type=int, default=100, help='Number of training iterations.')
    epf_parser.add_argument('--only_MRF', action='store_true', default=False, help='Freeze the pretrained GNN and train only MRF in M step.')
    args = parser.parse_args()
    args_dict = {}
    for group in parser._action_groups:
        if group.title == 'optional GMNN arguments':
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict['gmnn_args'] = argparse.Namespace(**group_dict)
        elif group.title == 'optional EPFGNN arguments':
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict['epf_args'] = argparse.Namespace(**group_dict)            
        else:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict.update(group_dict)
    return argparse.Namespace(**args_dict)

def name_model(fold, args):
    assert args.model in ['GCN', 'GAT', 'GMNN', 'EPFGNN'], f'Unexpected model name {args.model}.'
    name = args.model
    if args.model == 'EPFGNN' and args.epf_args.only_MRF: name += '_MRF'
    name += "_dp" + str(args.dropout_rate).replace(".","_") + "_"
    try:
        power =-math.floor(math.log10(args.wdecay))
        frac = str(args.wdecay)[-1] if power <= 4 else str(args.wdecay)[0]
        name += frac + "e_" + str(power)
    except:
        name += "0"
    name += "_f" + str(fold)
    return name

def metric_mean(result_dict):
    out = {}
    for key, val in result_dict.items():
        out[key] = np.mean(val) * 100
    return out

def metric_std(result_dict):
    out = {}
    for key, val in result_dict.items():
        out[key] = np.sqrt(np.var(val)) * 100
    return out

def create_nested_defaultdict(key_list):
    # To do: extend to *args
    out = {}
    for key in key_list:
        out[key] = defaultdict(list)
    return out

def save_prediction(predictions, name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    np.save(raw_dir/file_name, predictions)

def load_prediction(name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    return np.load(raw_dir / file_name)

def plot_reliabilities(
        reliabilities: Sequence[Reliability], title, saveto, bgcolor='w'):
    linewidth = 1.0

    confs = [(r[0] / (r[2] + torch.finfo().tiny)).cpu().numpy()
             for r in reliabilities]
    accs = [(r[1] / (r[2] + torch.finfo().tiny)).cpu().numpy()
            for r in reliabilities]
    masks = [r[2].cpu().numpy() > 0 for r in reliabilities]

    nonzero_counts = np.sum(np.asarray(masks, dtype=np.long), axis=0)
    conf_mean = np.sum(
        np.asarray(confs), axis=0) / (nonzero_counts + np.finfo(np.float).tiny)
    acc_mean = np.sum(
        np.asarray(accs), axis=0) / (nonzero_counts + np.finfo(np.float).tiny)
    acc_std = np.sqrt(
        np.sum(np.asarray(accs) ** 2, axis=0)
        / (nonzero_counts + np.finfo(np.float).tiny)
        - acc_mean ** 2)
    conf_mean = conf_mean[nonzero_counts > 0]
    acc_mean = acc_mean[nonzero_counts > 0]
    acc_std = acc_std[nonzero_counts > 0]

    fig, ax1 = plt.subplots(figsize=(2, 2), facecolor=bgcolor)
    for conf, acc, mask in zip(confs, accs, masks):
        ax1.plot(
            conf[mask], acc[mask], color='lightgray',
            linewidth=linewidth / 2.0, zorder=0.0)
    ax1.plot(
        [0, 1], [0, 1], color='black', linestyle=':', linewidth=linewidth,
        zorder=0.8)
    ax1.plot(
        conf_mean, acc_mean, color='blue', linewidth=linewidth, zorder=1.0)
    ax1.fill_between(
        conf_mean, acc_mean - acc_std, acc_mean + acc_std, facecolor='b',
        alpha=0.3, zorder=0.9)

    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    # ax1.legend(loc="lower right")
    ax1.set_title(title)
    plt.tight_layout()
    ax1.set_aspect(1)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(saveto, bbox_inches='tight', pad_inches=0)
