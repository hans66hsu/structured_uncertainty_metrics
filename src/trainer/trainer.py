import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer

# This script is used for "train_gmnn.py"

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

class Trainer(object):
    """
    Trainer for GMNN. Code adpated from (https://github.com/DeepGraphLearning/GMNN)
    """
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = get_optimizer('adam', self.parameters, 0.01, args.wdecay)

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer('adam', self.parameters, 0.01, args.wdecay)

    def update(self, inputs, edge_index, target, idx):
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs, edge_index)
        loss = self.criterion(logits[idx], target[idx])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, edge_index, target, idx):
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs, edge_index)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, inputs, edge_index, target, idx):
        self.model.eval()

        logits = self.model(inputs, edge_index)
        loss = self.criterion(logits[idx], target[idx])
        log_prob = F.log_softmax(logits, dim=-1)
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.sum(0)

        return loss.detach().item(), accuracy.detach().item(), log_prob.detach()

    def predict(self, inputs, edge_index, tau=1):

        self.model.eval()

        logits = self.model(inputs, edge_index) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
