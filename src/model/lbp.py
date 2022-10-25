import torch
from typing import Tuple
from torch import Tensor
from torch_geometric.data import Data

class LoopyBP:
    def __init__(
            self,
            unary_neg_energy: Tensor,
            pairwise_neg_energy: Tensor,
            edge_indices: Tensor,
            edge_scales: Tensor = None,
            iters: int = 100):
        self.unary_neg_energy = unary_neg_energy
        self.pairwise_neg_energy = pairwise_neg_energy
        self.edge_indices = edge_indices
        self.edge_scales = edge_scales
        # print("edge_scales", edge_scales.shape)
        self.iters = iters
        self.msg_n2f = self.init_msg_n2f()

    @property
    def pairwise(self) -> Tensor:
        if self.edge_scales is None:
            return self.pairwise_neg_energy.unsqueeze(0).expand(
                self.num_edges, self.num_classes, self.num_classes)
        else:
            # print("edge scales", self.edge_scales.shape, self.pairwise_neg_energy.shape)
            return self.edge_scales.view(-1, 1, 1) \
                * self.pairwise_neg_energy.unsqueeze(0)

    @property
    def st_indices(self) -> Tensor:
        return torch.cat(
            (self.edge_indices, self.edge_indices.flip(1)), dim=0)

    @property
    def s_indices(self) -> Tensor:
        return self.st_indices[:, 0]

    @property
    def t_indices(self) -> Tensor:
        return self.st_indices[:, 1]

    @property
    def num_nodes(self) -> int:
        return self.unary_neg_energy.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_indices.size(0)

    @property
    def num_classes(self) -> int:
        return self.unary_neg_energy.size(1)

    def flip_direction(self, t: Tensor) -> Tensor:
        return torch.cat((t[self.num_edges:], t[:self.num_edges]), dim=0)

    def init_msg_n2f(self) -> Tensor:
        unary = self.unary_neg_energy
        dtype = unary.dtype
        device = unary.device
        return torch.zeros(
            2 * self.num_edges, self.num_classes, dtype=dtype, device=device)

    def update_msg_f2n(self, msg_n2f: Tensor) -> Tensor:
        pairwise = self.pairwise
        # pairwise [2E, C, C]
        return torch.max(
            torch.cat([pairwise, pairwise], dim=0) + msg_n2f.unsqueeze(2),
            dim=1)[0]

    def update_msg_2n(self, msg_f2n: Tensor) -> Tensor:
        return self.unary_neg_energy.scatter_add(
            0,
            self.t_indices[:, None].expand(
                2 * self.num_edges, self.num_classes),
            msg_f2n)

    def update_msg_n2f(self, msg_f2n: Tensor) -> Tensor:
        msg_2n = self.update_msg_2n(msg_f2n)
        msg_n2f = msg_2n[self.s_indices] - self.flip_direction(msg_f2n)
        # normalize messages
        return torch.log_softmax(msg_n2f, dim=1)

    def run(self):
        msg_n2f = self.msg_n2f
        for _ in range(self.iters):
            msg_f2n = self.update_msg_f2n(msg_n2f)
            msg_n2f = self.update_msg_n2f(msg_f2n)
        self.msg_n2f = msg_n2f

    def node_marginals(self) -> Tensor:
        msg_2n = self.update_msg_2n(self.update_msg_f2n(self.msg_n2f))
        return torch.softmax(msg_2n, dim=1)

    def edge_marginals(self) -> Tensor:
        num_edges, num_classes = self.num_edges, self.num_classes
        msg_n2f = self.update_msg_n2f(self.update_msg_f2n(self.msg_n2f))
        neg_e = self.pairwise + msg_n2f[:num_edges].unsqueeze(2) + msg_n2f[num_edges:].unsqueeze(1)
        return torch.softmax(
            neg_e.view(num_edges, num_classes ** 2), dim=1
        ).view(num_edges, num_classes, num_classes)

    def node_logits(self) -> Tensor:
        return self.update_msg_2n(self.update_msg_f2n(self.msg_n2f))

    def edge_logits(self) -> Tensor:
        num_edges, num_classes = self.num_edges, self.num_classes
        msg_n2f = self.update_msg_n2f(self.update_msg_f2n(self.msg_n2f))
        # print(self.pairwise.shape, msg_n2f[:num_edges].shape, msg_n2f[num_edges:].shape, msg_n2f.shape)
        # if edge_scales: pairwise [4E, C, C] msg_n2f [4E C]
        # if not edge_scales: pairwsie [2E. C, C] msg_n2f [4E C]
        neg_e = self.pairwise + msg_n2f[:num_edges].unsqueeze(2) + msg_n2f[num_edges:].unsqueeze(1)
        return neg_e.view(num_edges, num_classes ** 2).view(num_edges, num_classes, num_classes)       