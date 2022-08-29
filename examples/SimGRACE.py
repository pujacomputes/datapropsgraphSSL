import argparse
import os
import random
import sys

sys.path.append("../")
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from LinearProc import LinearProc
from torch_geometric.nn import (
    GINConv,
    global_add_pool,
    global_mean_pool,
    GCNConv,
    SAGEConv,
    GATConv,
    global_max_pool,
)
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import ToUndirected
import numpy as np
from utils import get_augmentor, load_datasets

def nt_xentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
    )
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction="sum")
    return loss / (2 * N)


def make_gin_conv(input_dim, out_dim):
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
    )

#SimGRACE: https://github.com/junxia97/SimGRACE/blob/badf37130438416b094f7f58dbd8311123a3950b/unsupervised_TU/simgrace.py#L133
def gen_ran_output(data, model, vice_model, eta):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device='cpu'
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if "project" in name: #SHARED PROJECTOR!
            adv_param.data = param.data
        else:
            adv_param.data = param.data + eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)           
    _, z2 = vice_model(data.x, data.edge_index, data.batch)
    return z2


class GConv(torch.nn.Module):
    def __init__(
        self,
        num_layer,
        emb_dim,
        JK="last",
        drop_ratio=0,
        in_dim=10,
        gnn_type="gin",
        graph_pooling="mean",
    ):
        super(GConv, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                in_dim = in_dim
            else:
                in_dim = emb_dim
            if gnn_type == "gin":
                self.gnns.append(make_gin_conv(in_dim, emb_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "sage":
                self.gnns.append(SAGEConv(emb_dim))
            else:
                print("INVALID GNN")
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ### select pooling
        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool

        self.dr = torch.nn.Dropout(p=drop_ratio)
        project_dim = 160
        bottleneck_dim = 128
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, project_dim),
        )

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.gnns, self.batch_norms):
            z = conv(z, edge_index)
            z = bn(z)
            z = self.dr(F.relu(z))
            zs.append(z)
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(zs, dim=1)
        elif self.JK == "last":
            node_representation = zs[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in zs]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in zs]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        graph_representation = self.pool(node_representation, batch)
        return node_representation, graph_representation

    def get_embeddings(self, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                _, x = self.forward(x, edge_index, batch)
                ret.append(x.cpu())
                y.append(data.y.cpu())
        ret = torch.cat(ret)
        y = torch.cat(y)
        return ret, y


class Encoder(torch.nn.Module):
    def __init__(self, encoder, args):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.args = args
        self.augmentor = get_augmentor(args)

    def forward(self, graph_batch):
        with torch.no_grad():
            _, graph_1, graph_2 = self.augmentor(graph_batch)
        z1, g1 = self.encoder(graph_1.x, graph_1.edge_index, graph_1.batch)
        z2, g2 = self.encoder(graph_2.x, graph_2.edge_index, graph_2.batch)
        return z1, z2, g1, g2

    def forward_test(self, graph_1):
        z1, g1 = self.encoder(graph_1.x, graph_1.edge_index, graph_1.batch)
        return z1, g1


def train(encoder_model, vice_model, dataloader, optimizer,eta=1e-3):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data_0 = data.to("cuda")
        optimizer.zero_grad()
        g2 = gen_ran_output(data_0,encoder_model,vice_model,eta)
        _, g1 = encoder_model(data_0.x, data_0.edge_index, data_0.batch)
        
        g1 = encoder_model.project(g1) 
        g2 = vice_model.project(g2) 
        loss = nt_xentLoss(z1=g1, z2=g2, temperature=0.2)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data_0 = data.to("cuda")
            _, g0 = encoder_model(data_0.x, data_0.edge_index, data_0.batch)
            x.append(g0)
            y.append(data_0.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    LinProc_classifier = LinearProc(
        num_epochs=200, learning_rate=0.001, weight_decay=0.0, test_interval=20
    )
    result = LinProc_classifier(x, y, split)
    encoder_model.lineval_classifier = LinProc_classifier.classifier
    return result


class svc_dataset(InMemoryDataset):
    def __init__(
        self, root, data_list, transform=None, pre_transform=None, pre_filter=None
    ):
        self.data_list = data_list
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.transform = transform
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        if self.pre_filter is not None:
            self.data_list = [data for data in self.data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        torch.save(self.collate(self.data_list), self.processed_paths[0])

def arg_parser():
    parser = argparse.ArgumentParser(description="SimGrace")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--multiplier", default=1.0, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--aug_type", default="gga_sym", type=str, choices=['gga','caa','gga_sym','caa_sym'])
    parser.add_argument("--aug_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=237, type=int)
    parser.add_argument("--projector", action="store_true")
    parser.add_argument('--dataset_list', nargs='+', help='List of Dataset Pickles', required=True)
    parser.add_argument('--dataset_name',type=str, default="A-B-C-D-E-F",help='Name of Dataset')
    parser.add_argument('--dataset_root',type=str, default="../data",help='Saving Processed Dataset')
    parser.add_argument('--save_path',type=str, default="../logs",help='Saving Processed Dataset')
     
    args = parser.parse_args()
    return args
 
def main():
    device = torch.device("cuda")
    args = arg_parser()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print('=> Loading: ',args.dataset_name)
    data_obj_list = load_datasets(args.dataset_list)

    dataset = svc_dataset(
        root="{}/{}-{}".format(args.dataset_root, args.dataset_name,args.multiplier),
        pre_transform=ToUndirected(),
        pre_filter=None,
        data_list=data_obj_list,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    print("=> eta: ",args.eta)
    
    gconv = GConv(
        num_layer=5,
        emb_dim=32,
        in_dim=data_obj_list[0].x.shape[1],
        JK="concat",
        drop_ratio=0,
        gnn_type="gin",
        graph_pooling="mean",
    ).to(device)
    
    vice_gconv = GConv(
        num_layer=5,
        emb_dim=32,
        in_dim=data_obj_list[0].x.shape[1],
        JK="concat",
        drop_ratio=0,
        gnn_type="gin",
        graph_pooling="mean",
    ).to(device)
    
    targets = [
        args.dataset_name,
        "SimGRACE",
        args.aug_type,
        args.aug_ratio,
        args.projector,
        args.seed,
        args.eta
    ]
    targets = [str(t) for t in targets]
    print("=> Experiment Name: ", ",".join(targets))

    save_path = "{}/{}".format(args.save_path, args.dataset_name) 
    if not os.path.exists(save_path):
        os.mkdir(save_path) 

    ckpt_save_path = "{}/{}/ckpts".format(args.save_path, args.dataset_name) 
    if not os.path.exists(ckpt_save_path):
        os.mkdir(ckpt_save_path) 

    optimizer = Adam(
        gconv.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    test_result = test(gconv, dataloader)
    print(
        "Epoch 0, Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
            train_acc=test_result["train_acc"],
            valid_acc=test_result["valid_acc"],
            test_acc=test_result["test_acc"],
        )
    )
    print("*" * 30)
    with tqdm(total=args.epochs, desc="(T)",disable=False) as pbar:
        for epoch in range(0, args.epochs):
            loss = train(gconv, vice_gconv, dataloader, optimizer,eta=args.eta)
            pbar.set_postfix({"loss": loss})
            pbar.update()

    test_result = test(gconv, dataloader)
    print(
        "Final, Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
            train_acc=test_result["train_acc"],
            valid_acc=test_result["valid_acc"],
            test_acc=test_result["test_acc"],
        )
    )
  
    """
    Compute Accuracy on Other Style Ratios.
    """
    train_acc_list = [] 
    test_acc_list = [] 
    for multiplier in [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        mult_data_list = [d.replace(str(args.multiplier), str(multiplier)) for d in args.dataset_list] 
        data_obj_list = load_datasets(mult_data_list)
        dataset = svc_dataset(
            root="{}/{}".format(args.dataset_root, args.dataset_name.replace(str(args.multiplier),str(multiplier))),
            transform=torch.nn.Identity(),
            pre_transform=ToUndirected(),
            pre_filter=None,
            data_list=data_obj_list,
        )

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, shuffle=False
        )
        test_result = test(gconv, dataloader)
        print(
            "SVC Ratio: {mult}, Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
                mult=multiplier,
                train_acc=test_result["train_acc"],
                valid_acc=test_result["valid_acc"],
                test_acc=test_result["test_acc"],
            )
        )
        test_acc_list.append(np.round(test_result['test_acc'],4))
        train_acc_list.append(np.round(test_result['train_acc'],4))
    print("=> Train Avg: ", np.mean(train_acc_list))
    print("=> Test Avg: ", np.mean(test_acc_list))

    """
    Save Final Checkpoint.
    """
    targets = [
        args.dataset_name,
        "simgrace",
        args.aug_type,
        args.aug_ratio,
        args.projector,
        np.round(test_result["train_acc"], 4),
        np.round(test_result["valid_acc"], 4),
        np.round(test_result["test_acc"], 4),
        np.round(np.mean(train_acc_list),4),
        np.round(np.mean(test_acc_list),4)
    ]
 
    
    sep = ","
    targets = [str(t) for t in targets]
    save_str = sep.join(targets)
    with open("{}/simgrace_logs.csv".format(save_path), "a") as file:
        file.write("{}\n".format(save_str))

    # save ckpt!
    ckpt = {
        "encoder": gconv.state_dict(),
        "classifier": gconv.lineval_classifier.state_dict(),
        "results": test_result,
        "train_acc": test_result["train_acc"],
        "valid_acc": test_result["valid_acc"],
        "test_acc": test_result["test_acc"],
        "epoch":epoch,
    }
    
    save_name = "simgrace_{mult}_{aug_type}_{aug_ratio}_{proj}_{seed}.ckpt".format(
        mult=args.multiplier,
        aug_type=args.aug_type,
        aug_ratio=args.aug_ratio,
        proj=args.projector,
        seed=args.seed,
    )
        
    torch.save(obj=ckpt, f="{}/{}".format(ckpt_save_path,save_name))
if __name__ == "__main__":
    main()
