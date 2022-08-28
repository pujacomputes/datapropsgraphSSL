from GCL.augmentors.augmentor import Augmentor
import numpy as np
import torch
from torch_geometric.utils import to_undirected

def drop_edges_caa(edge_index, edge_motif, pe,edge_attr=None):
    """
    Expects undirected graphs.
    edge_index: from PyG object.
    edge_motif: edge attributes corresponding to Style vs. Content.
    (However, edge_motif could be any attribute.)
    pe: probability of dropping 
    """

    # top half edges
    top_half_idx = edge_index[0] > edge_index[1]

    # droppable edges in the top half
    dropable_top_half_mask = (edge_motif == -1) & top_half_idx
    dropable_top_half_idx = [
        enum for enum, i in enumerate(dropable_top_half_mask) if i == True
    ]
    permutable_edges = max(int(dropable_top_half_mask.sum() * pe), 1)

    # select edges to KEEP
    motif_edges_to_keep = np.random.choice(
        dropable_top_half_idx, size=permutable_edges, replace=False
    )
    motif_mask = torch.zeros_like(edge_index[0], dtype=torch.bool)
    motif_mask[motif_edges_to_keep] = True
    retained_edges_mask = ((edge_motif == 1) ^ top_half_idx) ^ motif_mask

    top_half_new = torch.vstack(
        (edge_index[0][retained_edges_mask], edge_index[1][retained_edges_mask])
    )
    if edge_attr is not None:
        new_edge_index,[new_edge_motif,new_edge_attr] = to_undirected(edge_index=top_half_new,edge_attr=[edge_motif[retained_edges_mask],edge_attr[retained_edges_mask]])
    else:
        new_edge_index,new_edge_motif = to_undirected(edge_index=top_half_new,edge_attr=[edge_motif[retained_edges_mask]])
        new_edge_attr = None 
    return new_edge_index, new_edge_motif,new_edge_attr 


class EdgeRemovingCAA(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemovingCAA, self).__init__()
        self.pe = pe

    def augment(self, g):
        pass

    def __call__(self, g):
        x, edge_index, edge_motif = g.x, g.edge_index, g.edge_motif
        edge_index, edge_motif,edge_attr = drop_edges_caa(
            edge_index, edge_motif=edge_motif, pe=self.pe,edge_attr=g.edge_attr
        )
        g.edge_index = edge_index
        g.edge_motif = edge_motif
        g.edge_attr = edge_attr 
        return g
