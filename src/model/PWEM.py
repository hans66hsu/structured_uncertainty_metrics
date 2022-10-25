import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch.nn as nn

""" modules that used to construct PWGNN"""
class BPLeafToRoot_edge_rezero(MessagePassing):
    # average redistribution + edge rezero
    def __init__(self):
        # super(BPLeafToRoot_edge_rezero, self).__init__(aggr='add', flow="source_to_target")
        super(BPLeafToRoot_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed, rezero):
        # here to solve the problem of overcounting, we forward distributed unary and binary energy
        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, rezero=rezero)

    def message(self, x_j, binary, rezero):

        N, C = x_j.shape
        messages = torch.logsumexp( (x_j.view(N, -1, 1) + rezero.view(-1, 1, 1) * binary), axis=1)
        return messages

    def update(self, aggr_out,x):

        log_z = torch.logsumexp((x + aggr_out),axis=1)
        # normalizer for every piece, i.e. for every node.
        return log_z

class PWLoss_edge_rezero(MessagePassing):
    # average redistribution version + edge_rezero
    def __init__(self):
        # super(PWLoss_edge_rezero, self).__init__(aggr='add', flow="source_to_target")
        super(PWLoss_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed, log_z_redistributed, q, rezero):
        # forward params needed to construct message and update
        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, edge_index_params, binary, q, rezero):

        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(x_j*q_j,axis=1) + rezero * torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q):
        # return the loss for every piece, and final loss need the summation
        result = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        return result

class MFUpdate_edge_rezero(MessagePassing):
    # edge rezero inference
    # update q
    def __init__(self):
        super(MFUpdate_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, q, edge_index, binary, unary, rezero):
        # here x represent the q
        return self.propagate(edge_index, size=(q.size(0), q.size(0)), x=q, binary=binary, unary=unary,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, binary, rezero):
        messages = rezero.view(-1,1) * torch.mm(x_j, binary)
        return messages

    def update(self, aggr_out, unary):
        # return the loss for every piece, and final loss need the summation
        return F.softmax(unary + aggr_out, dim=1)