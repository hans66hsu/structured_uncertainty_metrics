import os
import math
import random
import pickle
import gc
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(split, init, args):
    # Evaluation
    val_result = defaultdict(list)
    test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))
    for fold in range(max_fold):
        epochs = 2000
        lr = 0.01 #0.05
        model_name = name_model(fold, args)
        
        # Early stopping
        patience = 100
        vlss_mn = float('Inf')
        vacc_mx = 0.0
        vacc_early_model = None
        state_dict_early_model = None
        curr_step = 0
        best_result = {}

        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(dataset, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)
        
        # print(model)
        data = data.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask]) 
            loss.backward()
            optimizer.step()

            # Evaluation on traing and val set
            accs = []
            nlls = []
            briers = []
            eces = []
            with torch.no_grad():
                model.eval()
                logits = model(data.x, data.edge_index)
                log_prob = F.log_softmax(logits, dim=1).detach()

                for mask in [data.train_mask, data.val_mask]:
                    eval = NodewiseMetrics(log_prob, data.y, mask)
                    acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
                    accs.append(acc); nlls.append(nll); briers.append(brier); eces.append(ece)

                ### Early stopping
                val_acc = acc; val_loss = nll
                if val_acc >= vacc_mx or val_loss <= vlss_mn:
                    if val_acc >= vacc_mx and val_loss <= vlss_mn:
                        state_dict_early_model = copy.deepcopy(model.state_dict())
                        b_epoch = i
                        best_result.update({'log_prob':log_prob,
                                            'acc':accs[1],
                                            'nll':nlls[1],
                                            'bs':briers[1],
                                            'ece':eces[1]})
                    vacc_mx = np.max((val_acc, vacc_mx)) 
                    vlss_mn = np.min((val_loss, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break
        
        eval = NodewiseMetrics(best_result['log_prob'], data.y, data.test_mask)
        acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
        test_result['acc'].append(acc); test_result['nll'].append(nll); test_result['bs'].append(brier)
        test_result['ece'].append(ece)

        del best_result['log_prob']
        for metric in best_result:
            val_result[metric].append(best_result[metric])

        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        file_name = dir / (model_name + '.pt')
        torch.save(state_dict_early_model, file_name)
    return val_result, test_result


if __name__ == '__main__':
    args = arg_parse()
    assert args.model != "GMNN", "Please train GMNN using train_gmnn.py"
    assert args.model != "EPFGNN", "Please train EPFGNN using train_epfgnn.py"
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5


    val_total_result = {'acc':[], 'nll':[]}
    test_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            val_result, test_result = main(split, init, args)
            for metric in val_total_result:
                val_total_result[metric].extend(val_result[metric])
                test_total_result[metric].extend(test_result[metric])

    val_mean = metric_mean(val_total_result)
    test_mean = metric_mean(test_total_result)
    test_std = metric_std(test_total_result)
    print(f"Val  Accuracy: &{val_mean['acc']:.2f} \t" + " " * 8 +\
            f"NLL: &{val_mean['nll']:.4f}")
    print(f"Test Accuracy: &{test_mean['acc']:.2f}\pm{test_std['acc']:.2f} \t" + \
            f"NLL: &{test_mean['nll']:.4f}\pm{test_std['nll']:.4f}")

