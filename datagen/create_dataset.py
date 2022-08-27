import copy
import os
import networkx as nx
import random
import numpy as np
from motif_dict import MOTIFDICT
import glob 
from collections import defaultdict
import argparse
import torch_geometric as geom 
from scipy import sparse
import torch
import tqdm

"""
Here, we generate our synthetic dataset.
We label edges and nodes so that we can
perform oracle augmentation later.
-1: (style) background graph
1: (content) Motif.
"""

def attach_motif(random_base_graph, motif_edge_list, label='Motif', host_node=None):
    """
    Given the random graph, select a "host node"
    where we will attach the motif, and then 
    temporarily remove its neigbhoring edges.
    """
    N = len(random_base_graph.nodes())
    if host_node is None:
        host_cands = [k for k, v in random_base_graph.nodes(data=True) if v["label"] == 0]
        host_node = random.choice(host_cands)
    neighbors = list(random_base_graph.neighbors(host_node))
    for u in neighbors:
        random_base_graph.remove_edge(u, host_node)

    """
    We insert the host node into the original node list,
    so that the motif is connected through the random base graph
    through this the host, and edges maintain the correct structure. 
    """
    num_motif_nodes = np.array(motif_edge_list).max() 
    motif_nodes = [N + i for i in range(num_motif_nodes)]
    motif_nodes.insert(random.randint(0, 3), host_node)
    
    for u, v in motif_edge_list: 
        random_base_graph.add_edge(motif_nodes[u], motif_nodes[v])

    for u in motif_nodes:
        random_base_graph.nodes[u]["label"] = label

    # restore host_node edges
    for u in neighbors:
        v = random.choice(motif_nodes)
        random_base_graph.add_edge(u, v)
    return random_base_graph

def random_tree(n):
    g = nx.generators.trees.random_tree(n)
    for i in range(n):
        g.nodes[i]["label"] = -1
    return g

def random_ba(n,m=3):
    g = nx.generators.barabasi_albert_graph(n, m=m, seed=42)
    for i in range(n):
        g.nodes[i]["label"] = -1
    return g

"""
For a list of base graphs, fix a list of candidate host (attachment) nodes
"""
def get_host_cands(graph_list, motif_count):
    # this is necessary b/c motifs can be added multiple times
    candidate_size = np.sum(motif_count)
    host_nodes = np.zeros(candidate_size, dtype=int)
    idx = 0
    for g, m in zip(graph_list, motif_count):
        host_cands = [k for k, v in g.nodes(data=True) if v["label"] == -1]
        for _ in range(m):
            host_nodes[idx] = random.choice(host_cands)
            idx += 1
    return host_nodes

def add_to_list(graph_list, g, label):
    graph_num = len(graph_list) + 1
    for u in g.nodes():
        g.nodes()[u]["graph_num"] = graph_num
    g.graph["graph_num"] = graph_num
    graph_list.append((g, label))


def make_class_dataset(sample_size, motif_edge_list, background_graph='tree', multiplier=0.8):
    all_graphs = []
    random.seed(1)

    """
    Randomly select how many motifs are inserted into 
    each class sample. 
    """
    motif_count = np.random.randint(low=1, high=3, size=sample_size)
    num_motif_nodes = np.array(motif_edge_list).max()
    
    """
    determine size of background graph
    """
    random_graph_size = np.floor(motif_count * num_motif_nodes * multiplier).astype(int)
    eps = np.random.randint(low=-2, high=2, size=sample_size)
    random_graph_size = random_graph_size + eps
    random_graph_size[random_graph_size <= 1] = 2

    """
    Generate the list of background graphs 
    and corresponding host nodes.
    """
    base_graphs = []
    for e_num,s_a in enumerate(random_graph_size):
        if background_graph.lower() == 'ba':
            g = random_ba(n=s_a)
        elif background_graph.lower() == 'tree':
            g = random_tree(s_a)
        base_graphs.append(g)

    host_nodes = get_host_cands(base_graphs, motif_count)
    counter = 0

    """
    Attach the given motif to the background graph.
    """
    for g_a, m_count in zip(base_graphs, motif_count):
        for i in range(m_count):
            h_n = host_nodes[counter]
            g_a = attach_motif(random_base_graph = g_a, 
                motif_edge_list=motif_edge_list,
                label=1, #rename if desired.
                host_node=h_n)
            counter += 1
        add_to_list(all_graphs, g_a, 0)
    return all_graphs

def get_correct_edges(g):
    nodes_by_label = defaultdict(list)
    for u, data in g.nodes(data=True):
        nodes_by_label[data["label"]].append(u)
    edges = []
    for label, nodes in nodes_by_label.items():
        if label == "-1" or label == -1:
            continue
        edges.extend([(int(u), int(v)) for u, v in g.subgraph(nodes).edges()])
    return edges


def main():
    parser = argparse.ArgumentParser(description="dataset generation")
    parser.add_argument("--dataset", type=str, default="a1")
    parser.add_argument("--background_graph", type=str, default="tree",choices=['tree','ba'])
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument(
        "--multiplier", type=float, default=0.2, help="size of background wrt to motifs"
    )
    parser.add_argument("--output_path", type=str, default="../data")
    args = parser.parse_args()
    if args.dataset not in MOTIFDICT.keys():
        print("=> {} has not been defined! Exiting!".format(args.dataset))
        exit() 
    else: 
        motif_edge_list = MOTIFDICT[args.dataset]
    graphs = make_class_dataset(sample_size=args.sample_size,
        motif_edge_list=motif_edge_list,
        background_graph=args.background_graph,
        multiplier=args.multiplier)

    EXP_NAME = "{}_{}_{}".format(args.background_graph,args.dataset, args.multiplier)
    print("=> Dataset Name: ",EXP_NAME)
    output_path = "{}/{}_{}_{}".format(args.output_path, args.background_graph,args.dataset, args.multiplier)
    try:
        os.makedirs(output_path)
        print("=> Created Output Path: ", output_path)
    except:
        print("=> Exists!: ", output_path)
    print("=> Num Graphs: ", len(graphs))

    dataset_list = []
    max_nodes,num_nodes, num_edges = 0,0,0
    for G,_ in tqdm.tqdm(graphs):
        if max_nodes < G.number_of_nodes():
            max_nodes = G.number_of_nodes()
        num_nodes += G.number_of_nodes()
        num_edges += G.number_of_edges()

        correct_edges = get_correct_edges(G)
        edge_idx, edge_attr = geom.utils.from_scipy_sparse_matrix(
            sparse.csr_matrix(nx.to_numpy_array(G))
        )
        x = torch.ones(G.number_of_nodes(), 10)
        role_id = [n[1]["label"] for n in G.nodes(data=True)]
        for u, v, d in G.edges(data=True):
            if (int(u), int(v)) in correct_edges or (int(v), int(u)) in correct_edges:
                d["weight"] = 1 
            else:
                d["weight"] = -1

        G_adj = nx.to_numpy_array(G)
        _, edge_motif = geom.utils.from_scipy_sparse_matrix(sparse.csr_matrix(G_adj))
        assert edge_motif.shape[0] == edge_idx.shape[1], print(
            edge_motif.shape[0], edge_idx.shape[1]
        )

        d_1 = geom.data.Data(
            x=x.float(),
            edge_index=edge_idx,
            edge_attr=edge_attr.float(),
            edge_motif=edge_motif.float(),
            name=args.dataset,
            role_id=role_id,
        )
        dataset_list.append(d_1)

    num_nodes /= len(graphs)
    num_edges /= len(graphs)

    print("Avg. Num Nodes -- {}".format(num_nodes))
    print("Max Nodes -- {}".format(max_nodes))
    print("Avg. Num Edges {}".format(num_edges))

    save_name = "{PREFIX}/{exp}/data.pkl".format(PREFIX=args.output_path,exp=EXP_NAME)
    print("=> Saving: ", save_name)
    torch.save(dataset_list, save_name)

if __name__ == "__main__":
    main()