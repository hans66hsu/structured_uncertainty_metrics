import torch
from src.model.PWEM import PWLoss_edge_rezero
from src.trainer.trainer import get_optimizer

class Trainer_PWEM(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.criterion = PWLoss_edge_rezero()
        if args.epf_args.only_MRF:
            for p in self.model.gnn_backbone.parameters():
                p.requires_grad = False
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # for p in self.parameters:
        #     print(p.shape)
        self.optimizer = get_optimizer('adam', self.parameters, 0.01, args.wdecay)

    def update_edge_rezero(self, input_p, target_p, edge_index):
        self.model.train()
        self.optimizer.zero_grad()
        U, U_redistribution, B, B_redistribution, Z, R= self.model.get_energies(input_p, edge_index)
        loss_for_pieces = self.criterion(U_redistribution, edge_index, B_redistribution, Z, target_p, R)
        loss = -torch.mean(loss_for_pieces)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_loss_average_edge_rezero(self, inputs_p, target_p, edge_index, mask):
        self.model.eval()
        U, U_redistribution, B, B_redistribution, log_z_redistributed, R = self.model.get_energies(inputs_p, edge_index)
        loss_for_pieces = self.criterion(U_redistribution, edge_index, B_redistribution, log_z_redistributed,
                                            target_p, R)
        loss = -torch.mean(loss_for_pieces[mask])
        return loss.detach().item()

    def evaluate(self, input, target, mask):
        self.model.eval()
        preds = torch.max(input[mask], dim=1)[1]
        correct = preds.eq(target[mask]).double()
        accuracy = correct.sum() / mask.sum(0)
        return accuracy.detach().item()

