import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

HYP_MODELS = ["HEM"]


class BaseH(KGModel):
    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag_t = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag_t.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        t_init = torch.ones((self.sizes[2], 1), dtype=self.data_type)
        self.t = nn.Parameter(t_init, requires_grad=True)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def save(self,path):
        ent_embedding = self.entity.weight.cpu().detach().numpy()
        np.save(file=path, arr=ent_embedding)
        print("save finish!!!")

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            c = F.softplus(self.c[queries[:, 1]] * self.t[queries[:, 2]])
            tail = self.entity(queries[:, 2])
            rel1 = self.rel_diag_t(queries[:, 1])
            rel1 = rel1 * tail
            rel1 = expmap0(rel1, c)
            return rel1, self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class HEM(BaseH):

    def get_queries(self,queries):
        c = F.softplus(self.c[queries[:, 1]]*self.t[queries[:, 2]])
        head = self.entity(queries[:, 0])
        rel1 = self.rel_diag(queries[:, 1])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = rel1*head
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel, c)
        res2 = mobius_add(rel1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])  





