import argparse
import copy
import os
import random
import sys

sys.path.append("../")
import torch
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
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
from edge_removing_gga import EdgeRemovingGGA
import numpy as np
from utils import load_datasets

def make_gin_conv(input_dim, out_dim):
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
    )


class GConv(torch.nn.Module):
    def __init__(
        self,
        num_layer,
        emb_dim,
        JK="last",
        drop_ratio=0,
        input_dim=10,
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
                in_dim = input_dim 
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
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, project_dim),
        )

    # def forward(self, x, edge_index, edge_attr):
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
    
    def forward_test(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
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

def compute_invariance(encoder_model, dataloader,augmentor,REPEATS=30):
    encoder_model.eval()
    total_inv_score = []
    with torch.no_grad(): 
        for data in tqdm(dataloader):
            data = data.to('cuda')
            _,orig_rep = encoder_model.forward_test(data)
            inv_scores = torch.zeros(data.y.shape[0]).cuda()
            for r in range(REPEATS):

                data_aug = augmentor(copy.deepcopy(data))
                _, g_aug = encoder_model.forward(data_aug.x, data_aug.edge_index,data.batch)
                sim_vals = (orig_rep * g_aug).sum(dim=1) / (orig_rep.norm(dim=1)* g_aug.norm(dim=1)) 
                inv_scores += sim_vals
                del data_aug
            inv_scores = inv_scores / REPEATS
            total_inv_score.append(inv_scores)
    final_inv_score = torch.cat(total_inv_score).mean()
    return final_inv_score

def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data_0 = data.to("cuda")
            _, g0 = encoder_model.forward_test(data_0)
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
    parser = argparse.ArgumentParser(description="svc/invariance")
    parser.add_argument("--multiplier", default=4.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--aug_type", default="gga", type=str, choices=['gga_sym','caa_sym'])
    parser.add_argument("--aug_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=237, type=int)
    parser.add_argument("--projector", action="store_true")
    parser.add_argument("--ckpt",type=str,default="/usr/workspace/trivedi1/Fall2022/datapropsgraphSSL/logs/A-B-C-D-E-F/aagae_4.0_gga_sym_0.2_False_237_0.ckpt")
    parser.add_argument("--method",type=str)
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


    gconv = GConv(
        num_layer=5,
        emb_dim=32,
        JK="concat",
        drop_ratio=0,
        gnn_type="gin",
        graph_pooling="mean",
    ).to(device)

    net_ckpt = torch.load(args.ckpt)['encoder']
    gconv.load_state_dict(net_ckpt)
    print("=> Loaded State Dict!")
    print("=> Method: ",args.method)
    print("=> Base Multiplier: ",args.multiplier)
    targets = [
        args.dataset_name,
        args.method,
        args.aug_type,
        args.aug_ratio,
        args.projector,
        args.seed,
    ]

    targets = [str(t) for t in targets]
    print("=> Experiment Name: ", ",".join(targets))
    print("=> Loaded Checkpoint: ",args.ckpt) 

    save_path = "{}/{}".format(args.save_path, "svc") 
    if not os.path.exists(save_path):
        os.mkdir(save_path) 

    """
    Compute Accuracy and Invariance on Other Style Ratios.
    """
    train_acc_list = [] 
    test_acc_list = []
    invariance_list = []
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
        
        """
        Change the augmentor passed to compute invariance to 
        understand learned invariance to different augmentations.
        """
        inv_score = compute_invariance(encoder_model=gconv,dataloader=dataloader,augmentor= EdgeRemovingGGA(pe=0.5))  
        print(
            "SVC Ratio: {mult}, Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f} -- Inv: {inv:.4f}".format(
                mult=multiplier,
                train_acc=test_result["train_acc"],
                valid_acc=test_result["valid_acc"],
                test_acc=test_result["test_acc"],
                inv=inv_score.cpu().item()
            )
        )
        test_acc_list.append(np.round(test_result['test_acc'],4))
        train_acc_list.append(np.round(test_result['train_acc'],4))
        invariance_list.append(inv_score.cpu().item())
    print("=> Train Avg: ", np.mean(train_acc_list))
    print("=> Test Avg: ", np.mean(test_acc_list))
    print("=> Inv. Avg: ", np.mean(invariance_list))
    
    targets = [
        args.dataset_name,
        args.method,
        args.aug_type,
        args.aug_ratio,
        args.projector,
    ]

    """
    Save Accuracies and Invariances to Log Files.
    """  
    sep = ","
    test_targets_acc = targets + test_acc_list
    test_targets_acc= [str(t) for t in test_targets_acc]
    save_str = sep.join(test_targets_acc)
    with open("{}/svc_acc_logs.csv".format(save_path), "a") as file:
        file.write("{}\n".format(save_str))
    
    sep = ","
    test_targets_inv = targets + invariance_list 
    test_targets_inv = [str(t) for t in test_targets_inv]
    save_str = sep.join(test_targets_inv)
    with open("{}/svc_inv_logs.csv".format(save_path), "a") as file:
        file.write("{}\n".format(save_str))

if __name__ == "__main__":
    main()
