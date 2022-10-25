import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import degree
from src.model.PWEM import BPLeafToRoot_edge_rezero, MFUpdate_edge_rezero
from src.model.lbp import LoopyBP
from src.data.data_utils import get_dir_edge_idx, pairwise_edge_idx

def create_model(dataset, args):
    """
    Create model with hyperparameters according to the datasets
    """  
    num_layers = 2
    if args.model == 'GAT':
        num_hidden = 8
        attention_head = [8, 1]
    else:
        num_hidden = 64

    if args.model == 'GCN':
        return GCN(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=num_hidden,
                    drop_rate=args.dropout_rate, num_layers=num_layers)
    if args.model == 'GAT':
        return GAT(in_channels=dataset.num_features, num_classes = dataset.num_classes, num_hidden=num_hidden,
                    attention_head=attention_head, drop_rate=args.dropout_rate, num_layers=num_layers)
    elif args.model =='GMNN':
        return GMNN(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=num_hidden,
                    drop_rate=args.dropout_rate, num_layers=num_layers, train_mask=dataset.data.train_mask, 
                    target=dataset.data.y, args=args.gmnn_args)
    elif args.model == 'EPFGNN':
        # degree for p model
        row,_ = dataset.data.edge_index
        deg = degree(row, dataset.data.x.size(0), dtype=dataset.data.x.dtype).view(-1, 1) + 1 # here already add self-loop
        rezero_size = int(dataset.data.edge_index.shape[1]/2)
        return EPFGNN(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=num_hidden,
                    drop_rate=args.dropout_rate, num_layers=num_layers, train_mask=dataset.data.train_mask, 
                    target=dataset.data.y, deg=deg, rezero_size=rezero_size)
    raise AssertionError(f'Unexpected model name {args.model}.')

class EPFGNN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers, train_mask, target, deg, rezero_size):
        super(EPFGNN, self).__init__()
        self.gnn_backbone = GCN(in_channels=in_channels,
                                num_classes=num_classes,
                                num_hidden=num_hidden,
                                drop_rate=drop_rate,
                                num_layers=num_layers)
        self.register_buffer('train_mask', train_mask)
        self.register_buffer('target', target)
        self.register_buffer('deg', deg)
        self.up_BP = BPLeafToRoot_edge_rezero()
        self.inf = MFUpdate_edge_rezero()
        
        self.binary = torch.nn.Parameter((torch.randn(num_classes, num_classes) + torch.eye(num_classes))/num_classes,requires_grad=True)  # identity between every piece
        self.rezero_coefficients = torch.nn.Parameter(torch.zeros(rezero_size), requires_grad=True)

    def forward(self, x, edge_index):
        unary, _, binary, _, Z, R = self.get_energies(x, edge_index)
        self.have_run_loopy(edge_index, unary, binary, R)
        return self.lbp.node_logits()

    def get_energies(self, x, edge_index):
        edge_index = pairwise_edge_idx(edge_index)
        unary = self.gnn_backbone(x, edge_index)
        binary = (self.binary + self.binary.T)/2
        rezero_coefficients = self.rezero_coefficients.repeat(2)
        log_z_reditributed = self.up_BP(unary/(self.deg), edge_index, binary/2, rezero_coefficients)
        return unary, unary/(self.deg), binary, binary/2, log_z_reditributed, rezero_coefficients

    @torch.no_grad()
    def inference(self, target_p, x, edge_index):
        """
        Piecewise training inference for E step
        """
        self.eval()
        # need to check whether B need to force to be simatry
        U, U_redistributed, B, B_redistributed, Z, R = self.get_energies(x, edge_index)
        q = self.inf(target_p, edge_index, B, U, R)
        return q

    @torch.no_grad()
    def loopy_bp(self, edge_index, unary, binary, edge_scales):
        directed_edge_index = get_dir_edge_idx(edge_index)
        # make them positive
        unary, binary = -unary, -binary
        # clamp to training ground-truth labels by setting a huge unary energy for
        # the wrong cases
        gt = self.target[self.train_mask].view(-1, 1)
        gt_ue = torch.gather(unary[self.train_mask], 1, gt)  # save good unary
        # print("check unary", unary.mean(), unary.max(), unary.min())
        unary[self.train_mask] = 1e5  # discourage bad choices
        unary.scatter_(1, gt, gt_ue)  # recover good unary

        unary = unary - unary.amin(dim=1, keepdim=True)
        binary = binary - binary.min()
        unary.clamp_(max=20)
        return LoopyBP(-unary, -binary, directed_edge_index, self.rezero_coefficients, iters=100)

    def have_run_loopy(self, edge_index, unary, binary, edge_scales):
        if not hasattr(self, 'lbp'):
            self.lbp = self.loopy_bp(edge_index, unary, binary, edge_scales)
            self.lbp.run()            
    
    @torch.no_grad()
    def node_marginals(self, x, edge_index):
        unary, _, binary, _, Z, R = self.get_energies(x, edge_index)
        self.have_run_loopy(edge_index, unary, binary, R)
        return self.lbp.node_marginals()
    
    @torch.no_grad()
    def edge_marginals(self, x, edge_index):
        unary, _, binary, _, Z, R = self.get_energies(x, edge_index)
        self.have_run_loopy(edge_index, unary, binary, R)
        return self.lby.edge_marginals()

class GMNN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers, train_mask, target, args):
        super().__init__()
        self.drop_rate = drop_rate
        self.register_buffer('train_mask', train_mask)
        self.register_buffer('target', target)
        self.args = args
        self.gnnq = GCN(in_channels=in_channels, num_classes=num_classes, num_hidden=num_hidden,
                    drop_rate=drop_rate, num_layers=num_layers)
        self.gnnp = GCN(in_channels=num_classes, num_classes=num_classes, num_hidden=num_hidden,
                    drop_rate=drop_rate, num_layers=num_layers)
        self.num_classes = num_classes

    def forward(self, x, edge_index):
        """
        The foward pass is called only in the "calibration.py" for post-hoc calibration.
        Update functions of Q model and P model for EM training are defined in "train_gmnn.py".
        """
        if self.args.inference =='q':
            return self.gnnq(x, edge_index)
        elif self.args.inference == 'p':
            out_all = torch.empty(self.args.num_samples, x.size(0), self.num_classes, dtype=torch.float32, device=x.device)
            for i in range(self.args.num_samples):
                inputs_p = self.update_p_data(x, edge_index)
                out_all[i] = self.gnnp(inputs_p, edge_index)
            out_mean = out_all.mean(0)
            return out_mean

    def update_p_data(self, x, edge_index):
        logits = self.gnnq(x, edge_index) / self.args.tau
        preds = torch.softmax(logits, dim=-1).detach()
        inputs_p = torch.empty((x.size(0), preds.size(1)), dtype=torch.float32, device=x.device)
        if self.args.draw == 'exp':
            inputs_p.copy_(preds)
        elif self.args.draw == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        elif self.args.draw == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        if self.args.use_gold == 1:
            temp = torch.zeros(self.train_mask.sum(), preds.size(1)).type_as(inputs_p)
            temp.scatter_(1, torch.unsqueeze(self.target[self.train_mask], 1), 1.0)
            inputs_p[self.train_mask] = temp
        return inputs_p 

class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), GCNConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, edge_index):
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](x, edge_index)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, attention_head, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        attention_head = [1] + attention_head
        layer_list = []
        for i in range(len(self.feature_list)-1):
            concat = False if i == num_layers-1 else True 
            layer_list.append(["conv"+str(i+1), GATConv(self.feature_list[i]* attention_head[i], self.feature_list[i+1], 
                                                        heads=attention_head[i+1], dropout=drop_rate, concat=concat)])
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, edge_index):
        for i in range(len(self.feature_list)-1):
            x = F.dropout(x, self.drop_rate, self.training)
            x = self.layer_list["conv"+str(i+1)](x, edge_index)
            if i < len(self.feature_list)-2:
                x = F.elu(x)
        return x