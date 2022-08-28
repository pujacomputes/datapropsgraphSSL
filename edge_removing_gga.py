from GCL.augmentors.augmentor import Augmentor
import numpy as np
import torch
from torch_geometric.utils import to_undirected

"""
While edge_removing uses dropout adj to drop 
any edge with probability 20/%. 

This is function deterministically drops 20% percent 
of the graph, or at least one edge. 
(Important in the case of small graphs.)
"""

def drop_edges_gga(edge_index, edge_motif, pe,edge_attr=None):
    """
    Expects undirected graphs.
    edge_index: from PyG object.
    edge_motif: edge attributes corresponding to Style vs. Content.
    (However, edge_motif could be any attribute.)
    pe: probability of dropping 
    """
    #pull out the top half
    row,col = edge_index[0], edge_index[1]
    mask = row < col
    row_sub, col_sub = row[mask], col[mask]

    #drop at least one edge 
    permutable_edges = max(int(mask.sum() * pe),1)

    #randomly select top edges - permuted edges
    edges_to_keep = np.random.choice(np.arange(len(row_sub)),size=mask.sum().item() - permutable_edges,replace=False)

    row_sub = row_sub[edges_to_keep]
    col_sub = col_sub[edges_to_keep]

    top_half_new= torch.vstack([row_sub,col_sub])
    new_edge_index = to_undirected(edge_index =top_half_new)

    new_edge_motif = edge_motif[edges_to_keep]
    if edge_attr is not None:
        new_edge_attr = edge_attr[edges_to_keep]
    else:
        new_edge_attr = edge_attr

    if edge_attr is not None:
        new_edge_index,[new_edge_motif,new_edge_attr] = to_undirected(edge_index=top_half_new,edge_attr=[edge_motif[edges_to_keep],edge_attr[edges_to_keep]])
    else:
        new_edge_index,new_edge_motif = to_undirected(edge_index=top_half_new,edge_attr=[edge_motif[edges_to_keep]])
        new_edge_attr = None 
     
    return new_edge_index, new_edge_motif,new_edge_attr 



class EdgeRemovingGGA(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemovingGGA, self).__init__()
        self.pe = pe

    def augment(self, g):
        pass

    def __call__(self, g):
        x, edge_index, edge_motif = g.x, g.edge_index, g.edge_motif
        edge_index, edge_motif,edge_attr = drop_edges_gga(
            edge_index, edge_motif=edge_motif, pe=self.pe,edge_attr=g.edge_attr
        )
        g.edge_index = edge_index
        g.edge_motif = edge_motif
        g.edge_attr = edge_attr
        return g
