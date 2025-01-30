"""Implementation based on the template of ALIGNN."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from matformer.models.utils import RBFExpansion
from matformer.utils import BaseSettings
from matformer.features import angle_emb_mp
from torch_scatter import scatter
from matformer.models.transformer import MatformerConv
from matformer.models.global1 import GATGNN_GIM1_globalATTENTION
from torch_geometric.data.batch import Batch
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

class MatformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["matformer"]
    conv_layers: int = 5
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 128
    triplet_input_features: int = 40
    node_features: int = 128
    fc_layers: int = 1
    fc_features: int = 128
    output_features: int = 1
    node_layer_head: int = 4
    edge_layer_head: int = 4
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    post_fc_count = 1,

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: MatformerConfig = MatformerConfig(name="matformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )

        self.angle_lattice = config.angle_lattice
        if self.angle_lattice:  ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=config.edge_features,
                ),
                nn.Linear(config.edge_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=config.triplet_input_features,
                ),
                nn.Linear(config.triplet_input_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(config.node_features * 6, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(config.node_features * 2, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

        self.edge_init = nn.Sequential(  ## module not used
            nn.Linear(3 * config.node_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0)  ## module not used

        self.angle_init_layers = nn.Sequential(  ## module not used
            nn.Linear(120, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layers = nn.ModuleList(  ## module not used
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.edge_layer_head, edge_dim=config.node_features)
                for _ in range(config.edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        ##================================
        ## global attention initialization
        act = 'softplus',
        batch_norm = "True",
        batch_track_stats = "True",
        dropout_rate = 0.0,
        self.global_att_LAYER = GATGNN_GIM1_globalATTENTION(128, act, batch_norm, batch_track_stats, dropout_rate)
        ##================================
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid
        # self.pool="global_add_pool,
        self.post_lin_list = torch.nn.ModuleList()
        dim2=128
        output_dim=1
        for i in range(1):
                if i == 0:
                    lin = torch.nn.Linear(128, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
        self.lin_out = torch.nn.Linear(dim2, output_dim)

    # pool = "global_add_pool",
    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        # initial node features: atom feature network...

        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)

        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)



        # 应用全局注意力机制
        out_a = self.global_att_LAYER(node_features, data.batch, data.glob_feat)

        out_x  = node_features * out_a

        #Post-GNN dense layers

        out_x = getattr(torch_geometric.nn, 'global_add_pool')(out_x, data.batch)
        # for i in range(0, len(self.post_lin_list)):
        #     out_x = self.post_lin_list[i](out_x)
        #     out_x = getattr(F, 'softplus')(out_x)
        out = self.lin_out(out_x)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out


        # print("应用全局注意力机制:out_a", out_a.shape)  # ut_a torch.Size([60, 1])
        #
        # print("应用全局注意力机制:features", features.shape)  # torch.Size([60, 128])


        # #crystal-level readout
        # features = scatter(node_features, data.batch, dim=0, reduce="mean")
        # # print("scatter:", features.shape)  # scatter: torch.Size([2, 128])
        #
        # features = self.fc(out_x )
        #
        # out = self.fc_out(features)
        # if self.link:
        #     out = self.link(out)
        # if self.classification:
        #     out = self.softmax(out)
        # # print("yesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyes")
        #
        # return torch.squeeze(out)


