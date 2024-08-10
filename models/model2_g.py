import torch
import torch.nn as nn
import torch_geometric
from models import gps_layer1

class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=21, d_embed=20,hidden_dim=None,pos_enc_dim=None,in_feat_dropout=None,
        d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
        d_gcn=[166,166,166],
    ):
        super(Prot3DGraphModel, self).__init__()

        # self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(d_vocab, hidden_dim)
        self.linear_e = nn.Linear(d_edge, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        d_gcn_in = d_gcn[0]
        # self.embed = nn.Embedding(d_vocab, d_embed)
        # self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        # self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        self.layers = nn.ModuleList()
        for i in range(len(gcn_layer_sizes) - 1):            
            # layers.append((gps_layer1.GPSLayer(
            #         gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
            #     'data'
            # ))       
            self.layers.append(gps_layer1.GPSLayer(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in))              
        
        # self.gcn = torch_geometric.nn.Sequential(
        #     'data', layers)        
        self.pool = torch_geometric.nn.global_mean_pool
        self.other=1.0

    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        batch = data.batch
        h_pos = data.lap_enc
        e = data.edge_attr

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_pos.float())
        h = h + h_lap_pos_enc
        data.x= self.in_feat_dropout(h)

        data.edge_attr = self.linear_e(e)
        # data = self.gcn(data)
        for mod in self.layers:
            data = mod(data)
        # x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(data.x, data.batch)
        return x

class Drug3DModel(nn.Module):
    def __init__(self,
                 d_vocab=21,hidden_dim=None,pos_enc_dim=None,in_feat_dropout=None,
        d_emb=66,  d_edge=16,
        d_gcn=[166,166,166],
    ):
        super(Drug3DModel, self).__init__()
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(d_vocab, hidden_dim)
        self.linear_e = nn.Linear(d_edge, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        d_gcn_in = d_gcn[0]
        # # self.embed = nn.Embedding(d_emb, d_embed)
        # self.proj_node = nn.Linear(d_emb, d_gcn_in)
        # self.proj_edge = nn.Linear(d_edge, d_gcn_in)

        gcn_layer_sizes = [d_gcn_in] + d_gcn
        self.layers = nn.ModuleList()
        for i in range(len(gcn_layer_sizes) - 1):            
            # layers.append((gps_layer1.GPSLayer(
            #         gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
            #     'data'
            # ))       
            self.layers.append(gps_layer1.GPSLayer(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in))          
        
        # self.gcn = torch_geometric.nn.Sequential(
        #     'data', layers)        
        
        self.pool = torch_geometric.nn.global_mean_pool
        self.other=1.0

    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        batch = data.batch
        h_pos = data.lap_enc
        e = data.edge_attr

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_pos.float())
        h = h + h_lap_pos_enc
        data.x= self.in_feat_dropout(h)

        data.edge_attr = self.linear_e(e)

        # x = self.gcn(x, edge_index, edge_attr)
        # data = self.gcn(data)
        for mod in self.layers:
            data = mod(data)

        x = torch_geometric.nn.global_mean_pool(data.x, data.batch)
        return x


class DTAModel(nn.Module):
    def __init__(self,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[256, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(DTAModel, self).__init__()

        # self.drug_model = DrugGVPModel(
        #     node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
        #     edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
        # )
        self.drug_model = Drug3DModel()
        drug_emb_dim = 256

        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
       
        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.top_fc = self.get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, xd, xp):
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)

        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)

        x = torch.cat([xd, xp], dim=1)
        x = self.top_fc(x)
        return x
