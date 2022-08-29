import argparse
import copy
import os
import pdb
import random
import sys
import seaborn as sns
sns.set_style("darkgrid")

sys.path.append("../")
import torch
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
from edge_removing_caa import EdgeRemovingCAA 
import numpy as np
from utils import load_datasets
import pdb
import matplotlib.pyplot as plt

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
        input_dim=10,
        drop_ratio=0,
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

#pairwise N x N cosine similarity between rows of N x D matrix m)
def pairwise_cos(m):
    norm = (m*m).sum(1) ** 0.5
    norm_m = (m.T/norm).T
    sim = norm_m @ norm_m.T    
    return sim

#Input: N x N similarity matrix
#       N-dim label vector
def compute_sep(rep, labels):
    sim = pairwise_cos(rep)
    zero = torch.zeros(sim.shape).cuda()
    label_mat = labels.repeat(labels.shape[-1], 1) #N x N matrix
    same_label = (label_mat == label_mat.T) #i,j-th element True iff label[i] == label[j]
    max_inclass_sim = torch.max(torch.where(same_label, sim, zero), 1)[0]
    max_outofclass_sim = torch.max(torch.where(~same_label, sim, zero), 1)[0]
    sep_score = max_inclass_sim / max_outofclass_sim
    return sep_score    

#https://arxiv.org/pdf/2005.10242.pdf
def lalign(x, y, alpha=2):
    return (x-y).norm(dim=1).pow(alpha).mean()

def inv_sep_score(encoder_model, dataloader, augmentor,REPEATS=30,inv=True):
    """
    Specificy whether to use invariance of alignment using inv=True/False.
    """
    encoder_model.eval()
    total_inv_score = []
    total_sep_score = []
    orig_reps = []
    orig_norms = []
    inv_scores = []
    with torch.no_grad(): 
        for data in tqdm(dataloader):
            data = data.to('cuda')
            _,orig_rep = encoder_model.forward_test(data)
            orig_norm = orig_rep.norm(dim=1)
            sep_score = compute_sep(orig_rep, data.y)
            total_sep_score.append(sep_score)
            orig_reps.append(orig_rep)
            orig_norms.append(orig_norm)
            inv_scores = torch.zeros(data.y.shape[0]).cuda()
            for _ in range(REPEATS):
                data_aug = augmentor(copy.deepcopy(data))
                _, g_aug = encoder_model.forward(data_aug.x, data_aug.edge_index,data.batch)
                if inv:
                    sim_vals = (orig_rep * g_aug).sum(dim=1) / (orig_norm * g_aug.norm(dim=1)) 
                else: #compute alignment
                    sim_vals = lalign(orig_reps, g_aug)
                inv_scores += sim_vals
                del data_aug
            inv_scores = inv_scores / REPEATS
            total_inv_score.append(inv_scores)

    final_inv_scores = torch.cat(total_inv_score)
    final_sep_scores = torch.cat(total_sep_score)
    return final_inv_scores, final_sep_scores

def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data_0 = data.to("cuda")
            #pdb.set_trace()
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
    parser = argparse.ArgumentParser(description="Inv vs. Sep")
    parser.add_argument("--multiplier", default=4.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--aug_type", default="caa_sym", type=str, choices=['gga_sym','caa_sym'])
    parser.add_argument("--aug_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=237, type=int)
    parser.add_argument("--projector", action="store_true")
    parser.add_argument('--dataset_list', nargs='+', help='List of Dataset Pickles', required=True)
    parser.add_argument('--dataset_name',type=str, default="A-B-C-D-E-F",help='Name of Dataset')
    parser.add_argument('--dataset_root',type=str, default="../data",help='Saving Processed Dataset')
    parser.add_argument('--save_path',type=str, default="../logs",help='Saving Processed Dataset')
    parser.add_argument('--ckpt',type=str,default = '/usr/workspace/trivedi1/Fall2022/datapropsgraphSSL/logs/A-B-C-D-E-F-4.0/graphcl_4.0_gga_sym_0.2_False_237_59.ckpt')
    parser.add_argument('--method',type=str,default = 'graphcl')
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
        root="{}/{}".format(args.dataset_root, args.dataset_name),
        pre_transform=ToUndirected(),
        pre_filter=None,
        data_list=data_obj_list,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    gconv = GConv(
        num_layer=5,
        emb_dim=32,
        JK="concat",
        input_dim=10,
        drop_ratio=0,
        gnn_type="gin",
        graph_pooling="mean",
    ).to(device)

    net_ckpt = torch.load(args.ckpt)['encoder']
    gconv.load_state_dict(net_ckpt)
    print("=> Loaded State Dict!")
    print("=> Method: ",args.method)

    targets = [
        args.dataset_name,
        args.multiplier,
        args.method,
        args.aug_type,
        args.aug_ratio,
        args.projector,
        args.seed,
    ]
    targets = [str(t) for t in targets]
    print("=> Experiment Name: ", ",".join(targets))
    save_path = "{}/{}/inv_vs_sep".format(args.save_path, args.dataset_name) 
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # we augment in the loop. 
    # so it is faster to compute invariance
    test_result = test(gconv, dataloader)
    print(
        "Pretrained, Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
            train_acc=test_result["train_acc"],
            valid_acc=test_result["valid_acc"],
            test_acc=test_result["test_acc"],
        )
    )

    """
    Change the Augmentor to compute invariance
    with respect to a different augmentation.
    """
    if 'gga' in args.aug_type:
        augmentor = EdgeRemovingGGA(pe=0.2)
        cmap = 'Reds' 
    if 'caa' in args.aug_type:
        augmentor = EdgeRemovingCAA(pe=0.2) 
        cmap = 'Blues'
    pointwise_inv, pointwise_sep = inv_sep_score(encoder_model=gconv,dataloader=dataloader,augmentor=augmentor)
    print("=> Avg. Invariance: {0:.4f} -- Avg. Sep: {1:.4f}".format(pointwise_inv.mean().cpu().item(), pointwise_sep.mean().cpu().item()))
    np.save("{PREFIX}/{dataset_name}_{multiplier}_{method}_{augtype}_{aug_ratio}_inv_scores"\
            .format(PREFIX=save_path, dataset_name = args.dataset_name, multiplier=args.multiplier, method=args.method,\
             augtype = args.aug_type, aug_ratio = args.aug_ratio), pointwise_inv.cpu().data.numpy())

    np.save("{PREFIX}/{dataset_name}_{multiplier}_{method}_{augtype}_{aug_ratio}_sep_scores"\
            .format(PREFIX=save_path, dataset_name = args.dataset_name, multiplier=args.multiplier, method=args.method,\
             augtype = args.aug_type, aug_ratio = args.aug_ratio), pointwise_sep.cpu().data.numpy())
    sep=","
    targets = [
        args.dataset_name,
        args.multiplier,
        args.method,
        args.aug_type,
        args.aug_ratio,
        args.projector,
        args.seed,
        np.round(test_result['test_acc'],4),
        np.round(pointwise_inv.mean().cpu().item(),4), 
        np.round(pointwise_sep.mean().cpu().item(),4), 
    ]
    targets = [str(t) for t in targets]
    save_str = sep.join(targets)
    print("=> Invariance Sep. Summary: : ",save_str)
    with open("{PREFIX}/invariance_sep_logs.csv".format(PREFIX=save_path), "a") as file:
        file.write("{}\n".format(save_str))

    save_str = " ".join(targets[:-2])
    fig, ax = plt.subplots(1, figsize=(4,4))
    ax = sns.kdeplot(x=pointwise_inv.cpu().data.numpy(), y=pointwise_sep.cpu().data.numpy(), shade=False, fill = True, cmap =cmap, cbar=True, palette = 'Set2',ax=ax)
    plt.xlabel("Invariance")
    plt.ylabel("Separability")
    sns.despine()
    ax.set_title(save_str, loc='center', wrap=True)
    fig.savefig("{PREFIX}/{dataset_name}_{multiplier}_{method}_{augtype}_{aug_ratio}.pdf".format(PREFIX=save_path, 
        dataset_name = args.dataset_name, 
        multiplier=args.multiplier, 
        method=args.method,
        augtype = args.aug_type, 
        aug_ratio = args.aug_ratio))
if __name__ == "__main__":
    main()
