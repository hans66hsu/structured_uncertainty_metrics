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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

pre_patience = 50
patience = 10

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
    # _, val_acc, preds_test = trainer_q.evaluate(data.x, data.edge_index, data.y, data.test_mask)
    # print("q_acc", val_acc)

def update_p_data(trainer_q, data, gmnn_args):
    preds = trainer_q.predict(data.x, data.edge_index, gmnn_args.tau)
    inputs_p = torch.empty((data.x.size(0), preds.size(1)), dtype=torch.float32, device=data.x.device)
    target_p = torch.empty((data.x.size(0), preds.size(1)), dtype=torch.float32, device=data.x.device)
    if gmnn_args.draw == 'exp':
        inputs_p.copy_(preds)
        target_p.copy_(preds)
    elif gmnn_args.draw == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif gmnn_args.draw == 'smp':
        idx_lb = torch.multinomial(preds, 1).squeeze(1)
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    if gmnn_args.use_gold == 1:
        temp = torch.zeros(data.train_mask.sum(), preds.size(1)).type_as(target_p)
        temp.scatter_(1, torch.unsqueeze(data.y[data.train_mask], 1), 1.0)
        inputs_p[data.train_mask] = temp
        target_p[data.train_mask] = temp
    return inputs_p, target_p

def train_p(trainer_p, trainer_q, data, epoch, args):
    inputs_p, target_p = update_p_data(trainer_q, data, args)
    results = []
    idx_all = torch.ones_like(data.val_mask, device=data.val_mask.device).bool()
    # Add early stopping
    vlss_mn = float('Inf')
    vacc_mx = 0.0
    best_preds_test = 0.0
    for i in range(epoch):
        loss = trainer_p.update_soft(inputs_p, data.edge_index, target_p, idx_all)
        val_loss, val_acc, _ = trainer_p.evaluate(inputs_p, data.edge_index, data.y, data.val_mask)
        _, _, preds_test = trainer_p.evaluate(inputs_p, data.edge_index, data.y, data.test_mask)
    #     if val_acc >= vacc_mx or val_loss <= vlss_mn:
    #         if val_acc >= vacc_mx and val_loss <= vlss_mn:
    #             state = dict([('model', copy.deepcopy(trainer_p.model.state_dict())), ('optim', copy.deepcopy(trainer_p.optimizer.state_dict()))])
    #             best_preds_test = preds_test
    #         vacc_mx = max(val_acc, vacc_mx) 
    #         vlss_mn = min(val_loss, vlss_mn)
    #         curr_step = 0
    #     else:
    #         curr_step += 1
    #         if curr_step >= patience:
    #             break
    # trainer_p.model.load_state_dict(state['model'])
    # trainer_p.optimizer.load_state_dict(state['optim'])
    return preds_test, inputs_p

def update_q_data(trainer_p, inputs_p, data, gmnn_args):
    preds = trainer_p.predict(inputs_p, data.edge_index)
    target_q = torch.clone(preds)
    if gmnn_args.use_gold == 1:
        temp = torch.zeros(data.train_mask.sum(), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(data.y[data.train_mask], 1), 1.0)
        target_q[data.train_mask] = temp
    return target_q

def train_q(trainer_q, trainer_p, inputs_p, data, epoch, args):
    target_q = update_q_data(trainer_p, inputs_p, data, args)
    idx_all = torch.ones_like(data.val_mask, device=data.val_mask.device).bool()
    # Add early stopping
    vlss_mn = float('Inf')
    vacc_mx = 0.0
    best_preds_test = 0.0
    for _ in range(epoch):
        loss = trainer_q.update_soft(data.x, data.edge_index, target_q, idx_all)
        val_loss, val_acc, _ = trainer_q.evaluate(data.x, data.edge_index, data.y, data.val_mask)
        _, _, preds_test = trainer_q.evaluate(data.x, data.edge_index, data.y, data.test_mask)
    #     if val_acc >= vacc_mx or val_loss <= vlss_mn:
    #         if val_acc >= vacc_mx and val_loss <= vlss_mn:
    #             state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    #             best_preds_test = preds_test
    #         vacc_mx = max(val_acc, vacc_mx) 
    #         vlss_mn = min(val_loss, vlss_mn)
    #         curr_step = 0
    #     else:
    #         curr_step += 1
    #         if curr_step >= patience:
    #             break
    # trainer_q.model.load_state_dict(state['model'])
    # trainer_q.optimizer.load_state_dict(state['optim'])
    return preds_test

def main(split, init, args):
    # Evaluation
    p_result = defaultdict(list)
    q_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        pre_epoch, epoch = 200, 100
        model_name = name_model(fold, args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data
        data = data.to(device)

        model = create_model(dataset, args).to(device)
        trainer_q = Trainer(model.gnnq, args)
        trainer_p = Trainer(model.gnnp, args)

        # Pretrain Q model first
        pre_train(trainer_q, data, pre_epoch)
        # Optimize P model and Q model iteratively using EM
        for k in range(args.gmnn_args.iter):
            p_pred, inputs_p = train_p(trainer_p, trainer_q, data, epoch, args.gmnn_args)
            q_pred = train_q(trainer_q, trainer_p, inputs_p, data, epoch, args.gmnn_args)


        # P and Q model evaluation
        for pred, result_dict in zip([p_pred, q_pred], [p_result, q_result]):
            eval = NodewiseMetrics(pred, data.y, data.test_mask)
            acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
            result_dict['acc'].append(acc); result_dict['nll'].append(nll); result_dict['bs'].append(brier)
            result_dict['ece'].append(ece)

        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        #save p model
        file_name = dir / ("p_"+ model_name + '.pt')
        torch.save(model.gnnp.state_dict(), file_name)
        #save q model
        file_name = dir / ("q_"+ model_name + '.pt')
        torch.save(model.gnnq.state_dict(), file_name)
    return p_result, q_result


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5

    p_total_result = {'acc':[], 'nll':[]}
    q_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            p_result, q_result = main(split, init, args)
            for metric in p_total_result:
                p_total_result[metric].extend(p_result[metric])
                q_total_result[metric].extend(q_result[metric])

    p_mean = metric_mean(p_total_result)
    p_std = metric_std(p_total_result)
    q_mean = metric_mean(q_total_result)
    q_std = metric_std(q_total_result)

    print(f"P model(Structural) Accuracy: &{p_mean['acc']:.2f}\pm{p_std['acc']:.2f} \t" + \
            f"NLL: &{p_mean['nll']:.2f}\pm{p_std['nll']:.2f}")
    print(f"Q model(Node-wise) Accuracy: &{q_mean['acc']:.2f}\pm{q_std['acc']:.2f} \t" + \
            f"NLL: &{q_mean['nll']:.2f}\pm{q_std['nll']:.2f}")
