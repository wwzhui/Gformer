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

import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)




class GATGNN_GIM1_globalATTENTION(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(GATGNN_GIM1_globalATTENTION, self).__init__()
        self.act = act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        assert fc_layers > 1, "Need at least 2 fc layer"
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim+108, dim)
                self.global_mlp.append(lin)
            else:
                if i!= self.fc_layers :
                    lin = torch.nn.Linear(dim, dim)
                else:
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)
            if self.batch_norm == "True":
                # bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

    def forward(self, x, batch, glbl_x ):
        # 打印 x 和 glbl_x 的形状，方便检查问题
        # print(f"Shape of x: {x.shape}, Shape of glbl_x: {glbl_x.shape}")
        out = torch.cat([x,glbl_x],dim=-1)
        for i in range(0, len(self.global_mlp)):
            if i!= len(self.global_mlp) -1:
                out = self.global_mlp[i](out)
                out = getattr(F, 'softplus')(out)


            else:
                out = self.global_mlp[i](out)
                out = tg_softmax(out,batch)

        return out

    # out = getattr(F, self.act)(out)


