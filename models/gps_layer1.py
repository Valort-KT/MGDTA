import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch




class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, in_cha,out_cha,edge_dim, num_heads=1, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False,emb_dim = None,signet_dim=None):
        super().__init__()

        self.dim_h = out_cha
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]



        # self.local_model = torch_geometric.nn.TransformerConv(in_cha,out_cha,edge_dim)
        self.local_model = torch_geometric.nn.Sequential('x, edge_index, edge_attr', 
                                                         [(torch_geometric.nn.TransformerConv(
                    in_cha, out_cha, edge_dim=edge_dim),
                'x, edge_index, edge_attr -> x')])

        self.self_attn = torch.nn.MultiheadAttention(
                in_cha, num_heads, dropout=self.attn_dropout, batch_first=True)



        self.norm1_local = nn.BatchNorm1d(self.dim_h)
        self.norm1_attn = nn.BatchNorm1d(self.dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(self.dim_h, self.dim_h * 2)
        self.ff_linear2 = nn.Linear(self.dim_h * 2, self.dim_h)
        self.act_fn_ff = self.activation()

        self.norm2 = nn.BatchNorm1d(self.dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        
        h_local = self.local_model(x=h, edge_index=batch.edge_index,edge_attr=batch.edge_attr)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.

        h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

        # Multi-head attention.

        h_attn = self.self_attn(h,h,h)[0]


        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.

        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)

        h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

