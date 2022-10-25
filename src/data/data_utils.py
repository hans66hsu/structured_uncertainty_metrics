import os
import re
import numpy as np
import torch
from pathlib import Path
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid
from torch_geometric.io.planetoid import index_to_mask
from torch_geometric.transforms import NormalizeFeatures
from src.data.split import get_idx_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run at console -> python -c 'from src.data.data_utils import *; split_data("Cora", 5, 3, 85)'
def split_data(
        name: str, 
        samples_in_one_fold: int, 
        k_fold: int, 
        test_samples_per_class: int):
    """
    name: str, the name of the dataset
    samples_in_one_fold: int, sample x% of each class to one fold   
    k_fold: int, k-fold cross validation. One fold is used as validation the rest portions are used as training
    test_samples_per_class: int, sample x% of each class for test set
    """
    print(name)
    dataset = Planetoid(root='./data/', name=name, split='random')

    split_type = str(samples_in_one_fold)+"_"+str(k_fold)+'f_'+str(test_samples_per_class)       
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # For each configuration we split the data five times
    for i in range(5):
        assert int(samples_in_one_fold)*int(k_fold)+int(test_samples_per_class) <= 100, "Invalid fraction" 
        k_fold_indices, test_indices = get_idx_split(dataset,
                    samples_per_class_in_one_fold=samples_in_one_fold/100.,
                    k_fold=k_fold,
                    test_samples_per_class=test_samples_per_class/100.)
        split_file = f'{name.lower()}_split_{i}.npz'
        print(f"sample/fold/test: {len(k_fold_indices[0])}/{len(k_fold_indices)}/{len(test_indices)}")
        np.savez(raw_dir/split_file, k_fold_indices=k_fold_indices, test_indices=test_indices)

def load_data(name: str, split_type: str, split: int, fold: int) -> Dataset:
    """
    name: str, the name of the dataset
    split_type: str, format {sample per fold ratio}_{k fold}_{test ratio}. For example, 5_3f_85
    split: int, index of the split. In total five splits were generated for each dataset. 
    fold: int, index of the fold to be used as validation set. The rest k-1 folds will be used as training set.
    """
    transform = NormalizeFeatures()
    dataset = Planetoid(root='./data/', name=name, transform=transform)
    load_split_from_numpy_files(dataset, name, split_type, split, fold)
    return dataset

def load_split_from_numpy_files(dataset, name, split_type, split, fold):
    """
    load train/val/test from saved k-fold split files
    """
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    assert raw_dir.is_dir(), "Split type does not exist."
    split_file = f'{name.lower()}_split_{split}.npz'
    masks = np.load(raw_dir / split_file, allow_pickle=True)
    val_indices = masks['k_fold_indices'][fold]
    train_indices = np.concatenate(np.delete(masks['k_fold_indices'], fold, axis=0))
    test_indices = masks['test_indices']
    dataset.data.train_mask = index_to_mask(train_indices, dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(val_indices, dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_indices, dataset.data.num_nodes)

def both_in_mask_edge_index(edge_index, mask):
    """
    Return edge indices where both nodes are in the mask
    """
    edge_mask = mask[edge_index]
    edge_mask_both =torch.mul(edge_mask[0,:], edge_mask[1,:])
    return torch.stack((edge_index[0][edge_mask_both], edge_index[1][edge_mask_both]))

def not_both_in_mask_edge_index(edge_index, mask, global_mask):
    """
    Return edge indices where both nodes are in the global_mask, but not both
    in the mask, assumes mask represents a subgraph w.r.t. global_mask
    """
    edge_global_mask = global_mask[edge_index]
    edge_global_mask_both = torch.mul(
        edge_global_mask[0, :], edge_global_mask[1, :])
    edge_mask = mask[edge_index]
    edge_mask_both = torch.mul(edge_mask[0, :], edge_mask[1, :])
    edge_mask_not_both = torch.logical_xor(
        edge_global_mask_both, edge_mask_both)
    return torch.stack(
        (edge_index[0][edge_mask_not_both], edge_index[1][edge_mask_not_both]))

def ag_dis_edge_index(edge_index, label, agree_or_disagree):
    """
    Return edge indices where neighboring nodes are agreed/disagreed with each other
    """    
    edge_label = label[edge_index]
    agreed_mask = torch.eq(edge_label[0,:], edge_label[1,:])
    if agree_or_disagree == 'agree':
        return edge_index[:,agreed_mask]
    elif agree_or_disagree == 'disagree':
        disagreed_mask = ~agreed_mask
        return edge_index[:,disagreed_mask]
    raise AssertionError(f'agree_or_disagree needs to be "agree" or "disagre".')

def edge_index_to_node_index(num_nodes, edge_index):
    """
    Return a boolean mask where nodes that lie in the edge set are True
    """
    node_index = torch.unique(torch.cat((edge_index[0,:],edge_index[1,:])))
    node_mask = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_mask[node_index] = 1
    return node_mask.bool()

def get_masks(mask_name, data):
    if mask_name == 'Train':
        mask = data.train_mask
        global_mask = torch.logical_or(mask, data.val_mask)
    elif mask_name == 'Val':
        mask = data.val_mask
        global_mask = torch.logical_or(mask, data.train_mask)
    elif mask_name == 'Test':
        mask = data.test_mask
        global_mask = torch.logical_or(
            torch.logical_or(mask, data.train_mask), data.val_mask)
    else:
        raise ValueError("Invalid mask_name")
    return mask, global_mask

def get_edge_indices(edge_choice, data, mask, global_mask):
    if edge_choice == 'both':
        edge_indices = both_in_mask_edge_index(data.edge_index, mask)
    elif edge_choice == 'any':
        edge_indices = not_both_in_mask_edge_index(
            data.edge_index, torch.logical_xor(mask, global_mask), global_mask)
    elif edge_choice == 'full':
        edge_indices = both_in_mask_edge_index(data.edge_index, global_mask)
    else:
        raise ValueError(f'unrecognized edge choice: {edge_choice}')
    return edge_indices

def node2edge_label(label, directed_edge_index, num_classes):
    small_idx, large_idx = directed_edge_index[:,0], directed_edge_index[:,1] 
    small_lbl, large_lbl = label[small_idx], label[large_idx]
    return small_lbl * num_classes + large_lbl

def get_dir_edge_idx(edge_index: Tensor) -> Tensor:
    ei = edge_index
    return ei[:, ei[0] < ei[1]].T.contiguous()

def pairwise_edge_idx(edge_index: Tensor) -> Tensor:
    """
    Reorder edge indices according which source/target node has greater index
    """
    dir_ei = get_dir_edge_idx(edge_index)
    dir_ei = dir_ei.T.contiguous()
    rev_ei = torch.stack((dir_ei[1,:], dir_ei[0,:]))
    return torch.cat((dir_ei, rev_ei), 1)