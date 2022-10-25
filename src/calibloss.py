from typing import NamedTuple
import abc
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor, LongTensor, BoolTensor


# ref: https://stackoverflow.com/a/71801795
# do partial sums along dim 0 of tensor t
def partial_sums(t: Tensor, lens: LongTensor) -> Tensor:
    device = t.device
    elems, parts = t.size(0), len(lens)
    ind_x = torch.repeat_interleave(torch.arange(parts, device=device), lens)
    total = len(ind_x)
    ind_mat = torch.sparse_coo_tensor(
        torch.stack((ind_x, torch.arange(total, device=device)), dim=0),
        torch.ones(total, device=device, dtype=t.dtype),
        (parts, elems),
        device=device)
    return torch.mv(ind_mat, t)


class Reliability(NamedTuple):
    conf: Tensor
    acc: Tensor
    count: LongTensor


class ECE(nn.Module):
    binning_schemes = ('equal_width', 'uniform_mass')

    @staticmethod
    def equal_width_binning(
            confs: Tensor, corrects: BoolTensor, bins: int
    ) -> Reliability:
        sortedconfs, sortindices = torch.sort(confs)
        binidx = (sortedconfs * bins).long()
        binidx[binidx == bins] = bins - 1
        bincounts = binidx.bincount(minlength=bins)
        bincumconfs = partial_sums(sortedconfs, bincounts)
        bincumcorrects = partial_sums(
            corrects[sortindices].to(dtype=torch.get_default_dtype()),
            bincounts)
        return Reliability(
            conf=bincumconfs, acc=bincumcorrects, count=bincounts)

    @staticmethod
    def uniform_mass_binning(
            confs: Tensor, corrects: BoolTensor, bins: int
    ) -> Reliability:
        device = confs.device
        sortedconfs, sortindices = torch.sort(confs)
        indices = torch.div(
            torch.arange(bins + 1, device=device) * len(corrects),
            bins,
            rounding_mode='floor')
        bincounts = indices[1:] - indices[:-1]
        bincumconfs = partial_sums(sortedconfs, bincounts)
        bincumcorrects = partial_sums(
            corrects[sortindices].to(dtype=torch.get_default_dtype()),
            bincounts)
        return Reliability(
            conf=bincumconfs, acc=bincumcorrects, count=bincounts)

    def __init__(self, bins: int = 20, scheme: str = 'equal_width', norm=1):
        """
        bins: int, number of bins
        scheme: str, binning scheme
        norm: int or float, norm of error terms

        defaults follows:
        "On Calibration of Modern Neural Networks, Gou et. al., 2017"
        """
        assert scheme in ECE.binning_schemes
        super().__init__()
        self.bins = bins
        self.scheme = scheme
        self.norm = norm

    def binning(
            self, confs: Tensor, corrects: BoolTensor
    ) -> Reliability:
        scheme = self.scheme
        if scheme == 'equal_width':
            return ECE.equal_width_binning(confs, corrects, self.bins)
        elif scheme == 'uniform_mass':
            return ECE.uniform_mass_binning(confs, corrects, self.bins)
        else:
            raise ValueError(f'unrecognized binning scheme: {scheme}')

    def forward(self, confs: Tensor, corrects: BoolTensor) -> Tensor:
        bincumconfs, bincumcorrects, bincounts = self.binning(confs, corrects)
        # numerical trick to make 0/0=0 and other values untouched
        errs = (bincumconfs - bincumcorrects).abs() / (
            bincounts + torch.finfo().tiny)
        return ((errs ** self.norm) * bincounts / bincounts.sum()).sum()


class NodewiseBase(nn.Module, metaclass=abc.ABCMeta):
    # edge_index - shape: (2, E), dtype: long
    def __init__(self, node_index: LongTensor):
        super().__init__()
        self.node_index = node_index

    @abc.abstractmethod
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        raise NotImplementedError


class NodewiseNLL(NodewiseBase):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits = logits[self.node_index]
        nodegts = gts[self.node_index]
        return nnf.cross_entropy(nodelogits, nodegts)


class NodewiseBrier(NodewiseBase):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodeprobs = torch.softmax(logits[self.node_index], -1)
        nodeconfs = torch.gather(
            nodeprobs, -1, gts[self.node_index].unsqueeze(-1)).squeeze(-1)
        return (nodeprobs.square().sum(dim=-1) - 2.0 * nodeconfs
                ).mean().add(1.0)


class NodewiseECE(NodewiseBase):
    def __init__(
            self, node_index: LongTensor, bins: int = 15,
            scheme: str = 'equal_width', norm=1):
        super().__init__(node_index)
        self.ece_loss = ECE(bins, scheme, norm)

    def get_reliability(self, logits: Tensor, gts: LongTensor) -> Reliability:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs, nodepreds = torch.softmax(nodelogits, -1).max(dim=-1)
        nodecorrects = (nodepreds == nodegts)
        return self.ece_loss.binning(nodeconfs, nodecorrects)

    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs, nodepreds = torch.softmax(nodelogits, -1).max(dim=-1)
        nodecorrects = (nodepreds == nodegts)
        return self.ece_loss(nodeconfs, nodecorrects)


class EdgewiseBase(nn.Module, metaclass=abc.ABCMeta):
    # edge_index - shape: (2, E), dtype: long
    def __init__(self, edge_index: LongTensor):
        super().__init__()
        self.edge_index = edge_index

    @abc.abstractmethod
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        raise NotImplementedError


def edge_logprob_at(
        logits: Tensor, edge_index: Tensor, left_labels: Tensor,
        right_labels: Tensor
) -> Tensor:
    # (N, C) -> (E)
    leftindex, rightindex = edge_index
    left_logp = torch.log_softmax(logits[leftindex], -1)
    right_logp = torch.log_softmax(logits[rightindex], -1)
    left_logc = torch.gather(
        left_logp, -1, left_labels.unsqueeze(-1)).squeeze(-1)
    right_logc = torch.gather(
        right_logp, -1, right_labels.unsqueeze(-1)).squeeze(-1)
    edge_logc = left_logc + right_logc
    return edge_logc


def edge_conf(
        logits: Tensor, edge_index: Tensor
) -> Tensor:
        # (N, C) -> (E)
        leftindex, rightindex = edge_index
        left_p = torch.softmax(logits[leftindex], -1)
        right_p = torch.softmax(logits[rightindex], -1)
        diag_max = (left_p * right_p).amax(-1)
        off_diag_max = left_p.amax(-1) * right_p.amax(-1)
        return torch.maximum(diag_max, off_diag_max)


class EdgewiseNLL(EdgewiseBase):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        edgeindex = self.edge_index.flatten()
        edgelogits = logits[edgeindex]
        edgegts = gts[edgeindex]
        return nnf.cross_entropy(edgelogits, edgegts) * 2


class EdgewiseBrier(EdgewiseBase):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        leftindex, rightindex = self.edge_index
        left_p = torch.softmax(logits[leftindex], -1)
        right_p = torch.softmax(logits[rightindex], -1)
        edgeprobs = torch.einsum('el,em->elm', left_p, right_p)
        left_labels, right_labels = gts[leftindex], gts[rightindex]
        leftconfs = torch.gather(
            left_p, -1, left_labels.unsqueeze(-1)).squeeze(-1)
        rightconfs = torch.gather(
            right_p, -1, right_labels.unsqueeze(-1)).squeeze(-1)
        edgeconfs = leftconfs * rightconfs
        return (edgeprobs.square().sum(dim=(-2, -1)) - 2.0 * edgeconfs
                ).mean().add(1.0)

class EdgewiseECE(EdgewiseBase):
    def __init__(
            self, edge_index: LongTensor, bins: int = 15,
            scheme: str = 'equal_width', norm=1):
        super().__init__(edge_index)
        self.ece_loss = ECE(bins, scheme, norm)

    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        leftindex, rightindex = self.edge_index
        leftcorrects = (logits[leftindex].argmax(-1) == gts[leftindex])
        rightcorrects = (logits[rightindex].argmax(-1) == gts[rightindex])
        return self.ece_loss(
            edge_conf(logits, self.edge_index),
            leftcorrects * rightcorrects)

    def get_reliability(self, logits: Tensor, gts: LongTensor) -> Reliability:
        leftindex, rightindex = self.edge_index
        leftcorrects = (logits[leftindex].argmax(-1) == gts[leftindex])
        rightcorrects = (logits[rightindex].argmax(-1) == gts[rightindex])
        return self.ece_loss.binning(
            edge_conf(logits, self.edge_index),
            leftcorrects * rightcorrects)