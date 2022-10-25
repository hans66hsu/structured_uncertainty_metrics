import os
import math
import random
import copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch 
import torch.nn.functional as F
from src.model.model import create_model
from src.utils import set_global_seeds, arg_parse, name_model, metric_mean, metric_std
from src.metric import NodewiseMetrics
from src.data.data_utils import load_data
from src.trainer.trainer import Trainer
from src.trainer.trainer_pwem import Trainer_PWEM
import yaml

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

pre_patience = 50
patience = 10
em_patience = 5

def  load_experiment_configs(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        print(data)
    return data

def pre_train(trainer_q, data, epochs):
    one_hot_labels = F.one_hot(data.y % (max(data.y)+1)).float()
    # Add early stopping
    vlss_mn = float('Inf')
    vacc_mx = 0.0
    for _ in range(epochs):
        loss = trainer_q.update_soft(data.x, data.edge_index, one_hot_labels, data.train_mask)
        val_loss, val_acc, _ = trainer_q.evaluate(data.x, data.edge_index, data.y, data.val_mask)
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
            vacc_mx = max(val_acc, vacc_mx) 
            vlss_mn = min(val_loss, vlss_mn)
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= pre_patience:
                break
    trainer_q.model.load_state_dict(state['model'])
    trainer_q.optimizer.load_state_dict(state['optim'])

def init_target_p(trainer_q, data):
    preds = trainer_q.predict(data.x, data.edge_index)
    target_p = torch.empty((data.x.size(0), preds.size(1)), dtype=torch.float32, device=data.x.device)
    target_p.copy_(preds)

    # Same as use_gold in GMNN
    temp = torch.zeros(data.train_mask.sum(), preds.size(1)).type_as(target_p)
    temp.scatter_(1, torch.unsqueeze(data.y[data.train_mask], 1), 1.0)
    target_p[data.train_mask] = temp
    return target_p

def train_p(trainer_p, target_p, data, epoch, vacc_mx, state):
    # Add early stopping
    curr_step = 0
    for i in range(epoch):
        loss = trainer_p.update_edge_rezero(data.x, target_p, data.edge_index)
        val_loss = trainer_p.evaluate_loss_average_edge_rezero(data.x, target_p, data.edge_index, data.val_mask)
        temp_q = trainer_p.model.inference(target_p, data.x, data.edge_index)
        val_acc = trainer_p.evaluate(temp_q, data.y, data.val_mask)
        if val_acc > vacc_mx:
            state = dict([('model', copy.deepcopy(trainer_p.model.state_dict())), ('optim', copy.deepcopy(trainer_p.optimizer.state_dict()))])
            vacc_mx = val_acc
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    trainer_p.model.load_state_dict(state['model'])
    trainer_p.optimizer.load_state_dict(state['optim'])
    val_loss = trainer_p.evaluate_loss_average_edge_rezero(data.x, target_p, data.edge_index, data.val_mask)
    temp_q = trainer_p.model.inference(target_p, data.x, data.edge_index)
    val_acc = trainer_p.evaluate(temp_q, data.y, data.val_mask)
    return val_loss, val_acc, state

def update_target_p(trainer_p, target_p, data):
    preds = trainer_p.model.inference(target_p, data.x, data.edge_index)
    target_p = torch.clone(preds)
    temp = torch.zeros(data.train_mask.sum(), target_p.size(1)).type_as(target_p)
    temp.scatter_(1, torch.unsqueeze(data.y[data.train_mask], 1), 1.0)
    target_p[data.train_mask] = temp
    return target_p

def main(split, init, args):
    # Evaluation
    p_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        pre_epoch, epoch = 200, 50
        model_name = name_model(fold, args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data
        data = data.to(device)
        
        model = create_model(dataset, args).to(device)
        trainer_q = Trainer(model.gnn_backbone, args) 
        # Pretrain GNN first
        pre_train(trainer_q, data, pre_epoch)

        # Train the GNN backbone and MRF model end to end in M step
        trainer_p = Trainer_PWEM(model, args)
        # Initialize target p from GNN backbone
        target_p = init_target_p(trainer_q, data)

        # EM Algorithm
        vacc_mx, best_vacc_mx = 0.0, 0.0
        state = None
        curr_step = 0
        for k in range(args.epf_args.epf_iter):
            # M step update model parameters
            _, vacc_mx, state = train_p(trainer_p, target_p, data, epoch, vacc_mx, state)
            # E step update proposal distribution
            target_p = update_target_p(trainer_p, target_p, data)
            if vacc_mx > best_vacc_mx:
                best_vacc_mx = vacc_mx
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= em_patience:
                    break
        # Evaluation
        test_loss = trainer_p.evaluate_loss_average_edge_rezero(data.x, target_p, data.edge_index, data.test_mask)
        test_temp_q = trainer_p.model.inference(target_p, data.x, data.edge_index)
        test_acc = trainer_p.evaluate(test_temp_q, data.y, data.test_mask)
        p_result['acc'].append(test_acc); p_result['nll'].append(test_loss); 

        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        file_name = dir / (model_name + '.pt')
        torch.save(model.state_dict(), file_name)
    return p_result


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5

    p_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            p_result = main(split, init, args)
            for metric in p_total_result:
                p_total_result[metric].extend(p_result[metric])

    p_mean = metric_mean(p_total_result)
    p_std = metric_std(p_total_result)

    print(f"EPFGNN Accuracy: &{p_mean['acc']:.2f}\pm{p_std['acc']:.2f} \t" + \
            f"NLL: &{p_mean['nll']:.2f}\pm{p_std['nll']:.2f}")
