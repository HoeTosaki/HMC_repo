import torch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, DenseSAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch, to_dense_adj
import math
# import models.pool_modules.diffpool.diffpool_v2 as diffpool_v2
import hier_models.pool_modules.diffpool.diffpool as diffpool
import hier_models.pool_modules.mincut.mincutpool as mincutpool
import hier_models.pool_modules.cgipool.cgi_layer as cgipool

from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.pool import ASAPooling
from torch_geometric.nn.pool import graclus
from torch_geometric.nn.pool import MemPooling
# from torch_geometric.nn.dense.dmon_pool import DMoNPooling
# from torch_geometric.nn.dense.diff_pool import dense_diff_pool
# from torch_geometric.nn.dense.mincut_pool import dense_mincut_pool

from torch_scatter import scatter_sum,scatter_add,scatter_softmax,scatter_mean



from torch_geometric.utils import dense_to_sparse

class LocalPoolNet(nn.Module):
    def __init__(self, num_features, num_classes, gnn_hidden_dim,max_num_nodes,pooling_ratio,pool_name,pool_params,dropout=0.5,inner_cons=False):
        super(LocalPoolNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.gnn_hidden_dim = gnn_hidden_dim
        # self.mlp_hidden_dim = mlp_hidden_dim
        self.max_num_nodes = max_num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_name = pool_name
        self.dropout = dropout

        if self.is_dense(self.pool_name):
            if inner_cons:
                self.conv1 = DenseSAGEConv(self.num_features, self.gnn_hidden_dim)
            else:
                self.conv1 = SAGEConv(self.num_features, self.gnn_hidden_dim)
            self.conv2 = DenseSAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
            self.conv3 = DenseSAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
        else:
            self.conv1 = SAGEConv(self.num_features, self.gnn_hidden_dim)
            self.conv2 = SAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
            self.conv3 = SAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)

        num_nodes = [int(math.ceil(self.max_num_nodes * pow(self.pooling_ratio,i))) for i in range(4)]

        self.pool1 = self.create_pool(self.pool_name, in_nodes=num_nodes[0], out_nodes=num_nodes[1], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=pool_params)
        self.pool2 = self.create_pool(self.pool_name, in_nodes=num_nodes[1], out_nodes=num_nodes[2], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=pool_params)
        self.pool3 = self.create_pool(self.pool_name, in_nodes=num_nodes[2], out_nodes=num_nodes[3], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=pool_params)

        if not inner_cons:
            self.num_nodes = num_nodes
            self.pool_params = pool_params
            self.linear1 = torch.nn.Linear(self.gnn_hidden_dim * 2, self.gnn_hidden_dim)
            self.linear2 = torch.nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim // 2)
            self.linear3 = torch.nn.Linear(self.gnn_hidden_dim // 2, self.num_classes)

    def forward(self, data):
        '''
        :param data:
        :return:
        '''
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pool_loss = []

        if self.is_dense(self.pool_name):
            x = F.relu(self.conv1(x, edge_index)) # only the first conv is sparse
            x, mask = to_dense_batch(x, batch)
            adj = to_dense_adj(edge_index, batch)
            x, adj, loss = self.pool1(x, mask, adj)
            x1 = torch.cat([torch.max(x,dim=1)[0], torch.mean(x, dim=1)], dim=1)
            pool_loss.append(loss)


            x = F.relu(self.conv2(x, adj))
            x, adj, loss = self.pool2(x, None, adj)
            x2 = torch.cat([torch.max(x,dim=1)[0], torch.mean(x, dim=1)], dim=1)
            pool_loss.append(loss)

            x = F.relu(self.conv3(x, adj))
            x, adj, loss = self.pool3(x, None, adj)
            x3 = torch.cat([torch.max(x,dim=1)[0], torch.mean(x, dim=1)], dim=1)
            pool_loss.append(loss)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, batch, loss = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, batch, loss = self.pool2(x=x, edge_index=edge_index, batch=batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, batch, loss = self.pool3(x=x, edge_index=edge_index, batch=batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

        x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=1)

        return x,self.compute_pool_loss(self.pool_name,loss_lst=pool_loss)

    def create_pool(self,pool_name,in_nodes,out_nodes,in_dim,out_dim,pool_params):
        # currently support: 'DiffPool', 'MinCutPool', 'DiffPool_v2', 'GraClus', 'GMN', 'CGIPool'
        if pool_name == 'DiffPool':
            pool_params_def = {
                'dim_input': self.gnn_hidden_dim,
                'dim_embedding':self.gnn_hidden_dim,
                'current_num_clusters':in_nodes,
                'no_new_clusters':out_nodes,
                'pooling_type':'gnn',
                'invariant':False,
            }
            pool_params_def.update(pool_params)
            return diffpool.DiffPoolLayer_lpn(**pool_params_def)
        elif pool_name == 'MinCutPool':
            pool_params_def = {
                'hidden_dim': self.gnn_hidden_dim,
                'in_nodes': in_nodes,
                'out_nodes': out_nodes,
            }
            pool_params_def.update(pool_params)
            return mincutpool.MinCutPoolLayer_lpn(**pool_params_def)
        elif pool_name == 'CGIPool':
            pool_params_def = {
                'in_channels': self.gnn_hidden_dim,
                'ratio':self.pooling_ratio,
                'non_lin': torch.tanh,
            }
            pool_params_def.update(pool_params)
            return cgipool.CGIPool_lpn(**pool_params_def)
        elif pool_name == 'TopKPool':
            pool_params_def = {
                'in_channels':self.gnn_hidden_dim,
                'ratio':self.pooling_ratio,
                'min_score':None,
                'multiplier':1.,
                'nonlinearity':torch.tanh,
            }
            pool_params_def.update(pool_params)
            return TopKPool_lpn(**pool_params_def)
        elif pool_name == 'GMN':
            # MemPooling(in_channels=,out_channels=,heads=,num_clusters=,tau=)
            # TODO: Mem pooling.
            pass
        else:
            assert False, print(f'unmatched pool module name called {pool_name}')

    def is_dense(self,pool_name):
        if pool_name in ['DiffPool','MinCutPool']:
            # use sparse as first and dense for the others.
            return True
        elif pool_name in ['CGIPool','TopKPool']:
            return False
        else:
            assert False, print(f'unmatched pool module name called {pool_name}')

    def compute_pool_loss(self,pool_name,loss_lst):
        if pool_name == 'CGIPool':
            return 0.001 * sum(loss_lst) / 3
        elif pool_name == 'MinCutPool':
            return sum([mc+o for mc,o in loss_lst])
        elif pool_name == 'DiffPool':
            return sum([l + e for l, e in loss_lst])
        elif pool_name == 'TopKPool':
            return 0
        else:
            assert False, print(f'unmatched pool module name called {pool_name}')



class TopKPool_lpn(TopKPooling):
    def __init__(self,**kwargs):
        super(TopKPool_lpn, self).__init__(**kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        x, edge_index, edge_attr, batch, perm, score = super(TopKPool_lpn, self).forward(x, edge_index, edge_attr, batch, attn)
        return x, edge_index, batch, None


class MemPool_lpn(MemPooling):
    def __init__(self,**kwargs):
        super(MemPool_lpn, self).__init__(**kwargs)

    def forward(self, x, batch=None, mask=None):
        x,S = super(MemPool_lpn, self).forward(x,batch,mask)
        return x,
# x, edge_index, edge_attr, batch, perm, score[perm]


class LocalPoolLayer(nn.Module):
    def __init__(self, pool_name, in_nodes, in_dim, out_dim, out_nodes=None, pooling_ratio=None, verbose=False, pool_params=None):
        super().__init__()
        assert out_nodes is not None or pooling_ratio is not None
        assert not (out_nodes is not None and pooling_ratio is not None)
        self.pool_name = pool_name
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes if out_nodes is not None else int(math.ceil(self.in_nodes * pooling_ratio))
        self.pooling_ratio = pooling_ratio if pooling_ratio is not None else self.out_nodes / self.in_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pool_params = pool_params if pool_params is not None else dict()
        # self.edge_index_name = edge_index_name
        self.verbose = verbose
        assert self.pool_name in ['DiffPool', 'MinCutPool', 'DiffPool_v2', 'GraClus', 'GMN', 'CGIPool'], print(f'unsupported pool name {self.pool_name}')
        if verbose:
            print(f'use {self.is_dense_pool()} pooling layer [{self.pool_name}]')
        self.pool_layer = self.create_pool_layer()

    def create_pool_layer(self):
        # currently support: 'DiffPool', 'MinCutPool', 'DiffPool_v2', 'GraClus', 'GMN', 'CGIPool'
        if self.pool_name == 'DiffPool':
            pool_params_def = {
                'dim_input': self.in_dim,
                'dim_embedding': self.out_dim,
                'current_num_clusters': self.in_nodes,
                'no_new_clusters': self.out_nodes,
                'pooling_type': 'gnn',
                'invariant': False,
            }
            pool_params_def.update(self.pool_params)
            return diffpool.DiffPoolLayer_lpn(**pool_params_def)
        elif self.pool_name == 'MinCutPool':
            pool_params_def = {
                'hidden_dim': self.out_dim,
                'in_nodes': self.in_nodes,
                'out_nodes': self.out_nodes,
            }
            pool_params_def.update(self.pool_params)
            return mincutpool.MinCutPoolLayer_lpn(**pool_params_def)
        elif self.pool_name == 'CGIPool':
            pool_params_def = {
                'in_channels': self.out_dim,
                'ratio': self.pooling_ratio,
                'non_lin': torch.tanh,
            }
            pool_params_def.update(self.pool_params)
            return cgipool.CGIPool_lpn(**pool_params_def)
        elif self.pool_name == 'TopKPool':
            pool_params_def = {
                'in_channels': self.in_dim,
                'ratio': self.pooling_ratio,
                'min_score': None,
                'multiplier': 1.,
                'nonlinearity': torch.tanh,
            }
            pool_params_def.update(self.pool_params)
            return TopKPool_lpn(**pool_params_def)
        elif self.pool_name == 'GMN':
            # MemPooling(in_channels=,out_channels=,heads=,num_clusters=,tau=)
            # TODO: Mem pooling.
            pass
        else:
            assert False, print(f'unmatched pool module name called {self.pool_name}')

    def is_dense_pool(self):
        if self.pool_name in ['DiffPool','MinCutPool']:
            return True
        elif self.pool_name in ['CGIPool','TopKPool']:
            return False
        else:
            assert False, print(f'unmatched pool module name called {self.pool_name}')

    def forward(self, data, force_sparse=False,force_dense=False):
        if self.is_dense_pool():
            # assert current input is dense, unless force sparse.
            if force_sparse:
                x, edge_index, batch = data['x'], data['edge_index'], data['batch']
                x, mask = to_dense_batch(x, batch)
                adj = to_dense_adj(edge_index, batch)
            else:
                x,mask,adj = data['x'], data['mask'], data['adj']
            x, adj, loss = self.pool_layer(x, mask, adj)
            return x,adj,loss,mask
        else:
            # assert current input is dense, unless force dense.
            assert not force_dense # i cannot imagine by now the occasion when we have to convert dense to sparse...
            x, edge_index, batch = data['x'], data['edge_index'], data['batch']
            x, edge_index, batch, loss = self.pool1(x=x, edge_index=edge_index, batch=batch)
            return x,edge_index,loss,batch



class ChainPoolNet(LocalPoolNet):
    def __init__(self,**kwargs):
        kwargs['inner_cons'] = True
        super(ChainPoolNet, self).__init__(**kwargs)
        self.num_layers = 3
        # self.linear1 = torch.nn.Linear(self.gnn_hidden_dim * self.num_layers, self.gnn_hidden_dim)

        self.linear1 = torch.nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim)
        self.linear2 = torch.nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim)
        self.linear3 = torch.nn.Linear(self.gnn_hidden_dim, self.num_classes)

        self.linear1_inv = torch.nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim)
        self.linear2_inv = torch.nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim)
        self.linear3_inv = torch.nn.Linear(self.num_classes,self.gnn_hidden_dim)

        # self.pool1_inv = self.create_pool(self.pool_name, in_nodes=self.num_nodes[1], out_nodes=self.num_nodes[0], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=self.pool_params)
        # self.pool2_inv = self.create_pool(self.pool_name, in_nodes=self.num_nodes[2], out_nodes=self.num_nodes[1], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=self.pool_params)
        # self.pool3_inv = self.create_pool(self.pool_name, in_nodes=self.num_nodes[3], out_nodes=self.num_nodes[2], in_dim=self.gnn_hidden_dim, out_dim=self.gnn_hidden_dim, pool_params=self.pool_params)

        if self.is_dense(self.pool_name):
            self.conv1_inv = DenseSAGEConv(self.gnn_hidden_dim,self.num_features)
            self.conv2_inv = DenseSAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
            self.conv3_inv = DenseSAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
        else:
            self.conv1_inv = SAGEConv(self.gnn_hidden_dim,self.num_features)
            self.conv2_inv = SAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)
            self.conv3_inv = SAGEConv(self.gnn_hidden_dim, self.gnn_hidden_dim)

    def forward(self, data):
        '''
            we assume data has global defined chains & ees for extracting chain-level embeddings.
        :param data:
        :return:
        '''
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pool_loss = []
        adjs = []
        ss = []
        cs = []
        if self.is_dense(self.pool_name):
            # x = F.relu(self.conv1(x, edge_index))  # [BV,E]
            #
            # # TODO: combining the max. of c_emb.
            # c0 = F.normalize(data.chain_transform, p=1, dim=-1) # [B,V,C]
            # c_emb0 = self.extract_chain_emb(x,c0,batch) # [B,C,E]

            x, mask = to_dense_batch(x, batch) # x:[B,Vm,E]
            # Caution: mask is ignored later, since dense pooling will induce a completely dense graphs.
            adj = to_dense_adj(edge_index, batch) # [B,Vm,Vm]
            x = F.relu(self.conv1(x,adj,mask))
            c0 = F.normalize(data.chain_transform, p=1, dim=-1) # [BV,C]
            c_den0, _ = to_dense_batch(c0,batch) # [B,Vm,C]
            c_emb0 = tc.einsum('bve, bvc -> bce',[x,c_den0]) # [B,C,E]
            adjs.append(adj)
            x, adj, loss, s1 = self.pool1(x, mask, adj) # s:[B,V,V'], x:[B,V',E]
            pool_loss.append(loss)
            adjs.append(adj)
            ss.append(s1)
            cs.append(c0)
            # TODO: self.compute_chain_loss(s=s1,c=), chain assignment & pool assignment should be similar.

            c1 = tc.einsum('bij , bik -> bjk',[s1,c_den0]) # [B,V',C]
            c_emb1 = tc.permute(c1,dims=[0,2,1]) # [B,C,V']
            c_emb1 = F.normalize(c_emb1,dim=-1)
            c_emb1 = tc.einsum('bij, bjk -> bik',[c_emb1,x]) # [B,C,E]

            x = F.relu(self.conv2(x, adj))
            x, adj, loss, s2 = self.pool2(x, None, adj)
            pool_loss.append(loss)
            adjs.append(adj)
            ss.append(s2)
            cs.append(c1)

            c2 = tc.einsum('bij , bik -> bjk',[s2,c1]) # [B,V',C]
            c_emb2 = tc.permute(c2,dims=[0,2,1])# [B,C,V']
            c_emb2 = F.normalize(c_emb2,dim=-1)
            c_emb2 = tc.einsum('bij, bjk -> bik',[c_emb2,x]) # [B,C,E]

            x = F.relu(self.conv3(x, adj))
            x, adj, loss, s3 = self.pool3(x, None, adj)
            pool_loss.append(loss)
            adjs.append(adj)
            ss.append(s3)
            cs.append(c2)

            c3 = tc.einsum('bij , bik -> bjk',[s3,c2]) # [B,V',C]
            c_emb3 = tc.permute(c3,dims=[0,2,1])# [B,C,V']
            c_emb3 = F.normalize(c_emb3,dim=-1)
            c_emb3 = tc.einsum('bij, bjk -> bik',[c_emb3,x]) # [B,C,E]
            cs.append(c3)

        else:
            raise NotImplementedError
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, batch, loss = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, batch, loss = self.pool2(x=x, edge_index=edge_index, batch=batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, batch, loss = self.pool3(x=x, edge_index=edge_index, batch=batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            pool_loss.append(loss)

        # TODO: couple graph convs within each pool & residual learning at chain level between each pool. The resulting emb. will be sum with 1/x decay.
        # c_emb = tc.cat([c_emb0 + c_emb1 + c_emb2 + c_emb3],dim=-1) # [B,C,sum(E)]
        c_emb = c_emb3

        c_emb = F.relu(self.linear1(c_emb))
        c_emb = F.dropout(c_emb, p=self.dropout, training=self.training)
        c_emb = F.relu(self.linear2(c_emb))
        c_emb = F.relu(self.linear3(c_emb))

        return c_emb, self.compute_pool_loss(self.pool_name, loss_lst=pool_loss), adjs, ss,cs

    def inverse(self,c_emb,adjs,ss,cs):
        '''
            inverse chain emb into node-level kinematic params.
        :param c_emb: [B,C,E]
        :param adjs: [B,Vm,Vm]
        :param ss: [B,V,V']
        :param cs: [B,V',C]
        :return:
        '''
        # for lin in [self.linear3,self.linear2,self.linear1]:
        #     c_emb -= lin.bias
        #     c_emb = F.linear(c_emb,lin.weight.T,bias=None)
        for lin in [self.linear3_inv,self.linear2_inv,self.linear1_inv]:
            c_emb = F.relu(lin(c_emb))

        if self.is_dense(self.pool_name):
            s1,s2,s3 = ss
            c0,c1,c2,c3 = cs
            adj0,adj1,adj2,adj3 = adjs

            c3 = F.normalize(c3,p=1,dim=-1)
            x3 = tc.einsum('bce, bvc -> bve',[c_emb,c3])

            s3 = F.normalize(s3,p=1,dim=-1)
            x2 = tc.einsum('bje, bij -> bie',[x3,s3])
            x2 = F.relu(self.conv3_inv(x2,adj2))

            s2 = F.normalize(s2, p=1, dim=-1)
            x1 = tc.einsum('bje, bij -> bie', [x2, s2])
            x1 = F.relu(self.conv2_inv(x1,adj1))

            s1 = F.normalize(s1, p=1, dim=-1)
            x0 = tc.einsum('bje, bij -> bie', [x1, s1]) # x0:[B,Vm,E], c0:[B,V,C]
            x0 = F.relu(self.conv1_inv(x0,adj0))

        else:
            raise NotImplementedError

        return x0


    def extract_chain_emb(self,x,c,batch,p=1):
        # c = F.normalize(c, p=p, dim=-1)  # [BV,C]
        cx = c[:, :, None] * x[:, None, :]  # [BV,C,E] normalized CE.
        c_emb = scatter_add(cx, index=batch, dim=0)  # [B,C,E]
        return c_emb

    def compute_chain_loss(self,s,c,batch):
        '''
            TODO: require joints located on one chain share similar embeddings / assignments.
        :param s:
        :param c:
        :param batch:
        :return:
        '''
        pass

