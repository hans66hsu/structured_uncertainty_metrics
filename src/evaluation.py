import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from collections import defaultdict
from src.data.data_utils import load_data, ag_dis_edge_index, get_masks, get_edge_indices
from src.model.model import create_model
from src.metric import NodewiseMetrics, EdgewiseMetrics
from src.utils import set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
                        metric_mean, metric_std, save_prediction, plot_reliabilities
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def eval(data, log_prob, mask_name, eval_type, edge_choice: str='both'):
    mask, global_mask = get_masks(mask_name, data)
    eval_result = {}
    eval_edge_index = get_edge_indices(edge_choice, data, mask, global_mask)
    ag_edge_index = ag_dis_edge_index(eval_edge_index, data.y, 'agree')
    dis_edge_index = ag_dis_edge_index(eval_edge_index, data.y, 'disagree')
    if eval_type == 'Nodewise':
        eval = NodewiseMetrics(log_prob, data.y, mask)
    elif eval_type == 'Edgewise':
        eval = EdgewiseMetrics(log_prob, data.y, eval_edge_index)
    elif eval_type == 'Agree':
        eval = EdgewiseMetrics(log_prob, data.y, ag_edge_index)
    elif eval_type == 'Disagree':
        eval = EdgewiseMetrics(log_prob, data.y, dis_edge_index)
    else:
        raise ValueError('unknown eval_type')

    acc, nll, brier, ece = eval.acc(), eval.nll(), eval.brier(), eval.ece()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(split, init, eval_type_list, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    reliability_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        model_name = name_model(fold, args)
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = create_model(dataset, args).to(device)
        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        if args.model != "GMNN":
            file_name = dir / (model_name + '.pt')
            model.load_state_dict(torch.load(file_name))
        else:
            file_name = dir / ("p_"+ model_name + '.pt')
            model.gnnp.load_state_dict(torch.load(file_name))
            file_name = dir / ("q_"+ model_name + '.pt')
            model.gnnq.load_state_dict(torch.load(file_name))

        data = data.to(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            log_prob = F.log_softmax(logits, dim=1).detach()
        ### Store uncalibrated test result
        if args.save_prediction:
            save_prediction(log_prob.cpu().numpy(), args.dataset, args.split_type, split, init, fold, args.model, "uncal")

        for eval_type in eval_type_list:
            eval_result, reliability = eval(data, log_prob, 'Test', eval_type)
            reliability_result[eval_type].append(reliability)
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
            
        torch.cuda.empty_cache()
 
    return uncal_test_result, reliability_result



if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise', 'Edgewise', 'Agree', 'Disagree']
    max_splits,  max_init = 5, 5

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    reliability_all = defaultdict(list)
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            (uncal_test_result,
             reliability_result) = main(split, init, eval_type_list, args)
            for eval_type, eval_metric in uncal_test_result.items():
                reliability_all[eval_type].extend(reliability_result[eval_type])
                for metric in eval_metric:
                    uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                
    # print results
    for eval_type in eval_type_list:
        test_mean = metric_mean(uncal_test_total[eval_type])
        test_std = metric_std(uncal_test_total[eval_type])
        print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                            f"NLL: &{test_mean['nll']:.2f}$\pm${test_std['nll']:.2f} \t" + \
                            f"Brier: &{test_mean['bs']:.2f}$\pm${test_std['bs']:.2f} \t" + \
                            f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f}")
        if args.reli_diag:
            title = args.model + " " + eval_type 
            plot_reliabilities(
                reliability_all[eval_type], title,
                f'plots/{eval_type}_reliability_{args.dataset}_{args.model}.pdf')

