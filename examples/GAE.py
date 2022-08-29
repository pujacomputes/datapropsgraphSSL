import argparse
import math
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
from torch_geometric.utils import batched_negative_sampling
from utils import get_augmentor, load_datasets


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


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


class gae_decoder(torch.nn.Module):
    def __init__(self, hidden_dim=160):
        super(gae_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction="none")

        self.pool = global_add_pool
        self.ste = StraightThroughEstimator()

    def forward(self, x, pos_edge_index, neg_edge_index, batch):
        i_idx = torch.cat([pos_edge_index[0], neg_edge_index[0]])
        j_idx = torch.cat([pos_edge_index[1], neg_edge_index[1]])
        pos_size = len(pos_edge_index[0])
        neg_size = len(neg_edge_index[0])

        pred = self.decoder((x[i_idx] * x[j_idx]))
        pred = self.ste(pred)

        pos_pred, neg_pred = torch.split(pred, [pos_size, neg_size])
        pos_target = torch.ones_like(pos_pred).to(x.device)
        neg_target = torch.zeros_like(neg_pred).to(x.device)
        pos_loss = torch.nn.BCELoss(reduction="none")(pos_pred, pos_target)
        neg_loss = torch.nn.BCELoss(reduction="none")(neg_pred, neg_target)

        loss_pos = self.pool(pos_loss, batch[pos_edge_index[0]])
        loss_neg = self.pool(neg_loss, batch[neg_edge_index[0]])
        try:
            loss_rec = loss_pos + loss_neg
            loss = loss_rec.mean()
        except:
            loss = loss_pos.mean() + loss_neg.mean()
        return loss


class gae(torch.nn.Module):
    def __init__(self, encoder, decoder, args):
        super(gae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, graph_1, graph_2=None):
        x, _ = self.encoder(graph_1.x, graph_1.edge_index, graph_1.batch)
        with torch.no_grad():
            neg_sample_edge_index = batched_negative_sampling(
                graph_1.edge_index, batch=graph_1.batch
            )
        loss = self.decoder(x, graph_1.edge_index, neg_sample_edge_index, graph_1.batch)
        return loss

    def forward_test(self, data):
        x, g = self.encoder(data.x, data.edge_index, data.batch)
        return x, g


def train_gae(
    gae_model,
    dataloader,
    optimizer,
):

    gae_model.train()
    epoch_loss = 0
    for enum, data in enumerate(dataloader):
        optimizer.zero_grad()
        data_0 = data[0].to("cpu")  # original sample only!
        loss = gae_model.forward(data_0, None)
        if not math.isfinite(loss.item()):
            print("*" * 50)
            print("Loss is {}, stopping training".format(loss.item(), force=True))
            print("* " * 50)
            return
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def test(gae_model, dataloader):
    gae_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data_0 = data[0].to("cpu")
            _, g0 = gae_model.forward_test(data_0)
            x.append(g0)
            y.append(data[0].y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    LinProc_classifier = LinearProc(
        num_epochs=200, learning_rate=0.001, weight_decay=0.0, test_interval=20
    )
    result = LinProc_classifier(x, y, split)
    gae_model.lineval_classifier = LinProc_classifier.classifier
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
    parser = argparse.ArgumentParser(description="GAE")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--multiplier", default=4.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--aug_type", default="gga", type=str, choices=['gga_sym','caa_sym'])
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
    device = torch.device("cpu") #intentional. 
    args = arg_parser()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print('=> Loading: ',args.dataset_name)
    data_obj_list = load_datasets(args.dataset_list)

    cust_aug = get_augmentor(args)
    dataset = svc_dataset(
        root="{}/{}-{}".format(args.dataset_root, args.dataset_name,args.multiplier),
        transform=cust_aug,
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
        input_dim=dataset[0][0].num_features,
        drop_ratio=0,
        gnn_type="gin",
        graph_pooling="mean",
    ).to(device)

    gdecoder = gae_decoder(hidden_dim=160)
    encoder_model = gae(encoder=gconv, decoder=gdecoder, args=args).to(device)

    targets = [
        args.dataset_name,
        "GAE",
        args.aug_type,
        args.aug_ratio,
        args.projector,
        args.seed,
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
        encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print("=> Init Accuracy!")
    test_result = test(encoder_model, dataloader)
    print(
        "Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
            train_acc=test_result["train_acc"],
            valid_acc=test_result["valid_acc"],
            test_acc=test_result["test_acc"],
        )
    )

    losses = []
    with tqdm(total=args.epochs, desc="(T)") as pbar:
        for epoch in range(0, args.epochs):
            loss = train_gae(encoder_model, dataloader, optimizer)
            losses.append(loss)
            pbar.set_postfix({"loss": loss})
            pbar.update()
            test_result = test(encoder_model, dataloader)
            print(
                "Epoch: {epoch} -- Train: {train_acc:.4f} -- Valid: {valid_acc:.4f} -- Test: {test_acc:.4f}".format(
                    epoch=epoch,
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
        test_result = test(encoder_model, dataloader)
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
    # datasetname, aug_type, aug_ratio,train_acc, valid_acc,test_acc
    


    """
    Save Final Checkpoint.
    """
    targets = [
        args.dataset_name,
        "gae",
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
    with open("{}/gae_logs.csv".format(save_path), "a") as file:
        file.write("{}\n".format(save_str))

    # save ckpt!
    ckpt = {
        "encoder": encoder_model.encoder.state_dict(),
        "classifier": encoder_model.lineval_classifier.state_dict(),
        "results": test_result,
        "train_acc": test_result["train_acc"],
        "valid_acc": test_result["valid_acc"],
        "test_acc": test_result["test_acc"],
        "epoch":epoch,
        "losses":losses
    }
    
    save_name = "gae_{mult}_{aug_type}_{aug_ratio}_{proj}_{seed}.ckpt".format(
        mult=args.multiplier,
        aug_type=args.aug_type,
        aug_ratio=args.aug_ratio,
        proj=args.projector,
        seed=args.seed,
    )
        
    torch.save(obj=ckpt, f="{}/{}".format(ckpt_save_path,save_name))
if __name__ == "__main__":
    main()
