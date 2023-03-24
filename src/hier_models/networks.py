import torch
from torch_scatter import scatter_max, scatter_sum, scatter_softmax
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from src.hier_models.basic_modules import MLP, GCUTPL
# from src.hier_models.lpn import LocalPoolLayer
from torch_geometric.nn import GCNConv, SAGEConv, DenseSAGEConv
from torch_geometric.utils import to_dense_batch, to_dense_adj


class HierPool(torch.nn.Module):
    def __init__(self,type='naive'):
        super(HierPool, self).__init__()
        self.type = type
        if self.type == 'naive':
            pass # there's no learnable module in naive hierpool.
        else:
            raise NotImplementedError
    def forward(self,x,A_inv):
        # x:[BV,X], A_inv:[BV',V] caution for sparse matrix.
        return torch.matmul(A_inv,x)

class HierMeshEncoder(torch.nn.Module):
    def __init__(self,num_layers=4,hidden_dims=[128],pool_type='naive',verbose=False):
        super(HierMeshEncoder, self).__init__()
        self.num_layers = num_layers
        self.pool_type = pool_type
        if isinstance(hidden_dims,int):
            self.hidden_dims = [hidden_dims]*self.num_layers
        elif isinstance(hidden_dims,list) or isinstance(hidden_dims,tuple):
            assert len(hidden_dims) > 0
            if len(hidden_dims) < self.num_layers:
                self.hidden_dims = hidden_dims + [hidden_dims[-1]]*(self.num_layers - len(hidden_dims))
        self.module_dict = torch.ModuleDict()
        for i in range(self.num_layers):
            self.module_dict[f'gcn_{i}'] = GCUTPL(in_channels=hidden_dims[i-1] + 6 if i != 0 else 6, out_channels=hidden_dims[i], aggr='max')
            if i != self.num_layers - 1:
                self.module_dict[f'pool_{i}'] = HierPool(type=self.pool_type)
        self.module_dict['mlp'] = MLP([hidden_dims[-1],hidden_dims[-1]])
        self.verbose = verbose
        if self.verbose:
            print(f'HierMesh Encoder [hidden_dims {self.hidden_dims}]')
        self._gcns = None
        self._pools = None

    @property
    def gcns(self):
        if self._gcns is None:
            self._gcns = [self.module_dict[f'gcn_{i}'] for i in range(self.num_layers)]
        return self._gcns

    @property
    def pools(self):
        if self._pools is None:
            self._pools = [self.module_dict[f'pool_{i}'] for i in range(self.num_layers - 1)]
        return self._pools

    def forward(self,x0s,tpl_edge_indices,As,A_invs,batch,with_layer_xs=False):
        '''
            x0 - (3+3), vertex position & its normal.
        '''
        layer_xs = []
        x = x0s[0]
        for i in range(self.num_layers):
            x = self.module_dict[f'gcn_{i}'](x,tpl_edge_indices[i])
            x = ReLU(x)
            layer_xs.append(x)
            if i != self.num_layers:

                x = self.module_dict[f'pool_{i}'](x,A_invs[i])
        out_x = self.module_dict['mlp'](x)
        if not with_layer_xs:
            return out_x
        return out_x,layer_xs

class HierMeshDecoder(torch.nn.Module):
    def __init__(self,hme:HierMeshEncoder):
        super(HierMeshDecoder, self).__init__()
        self.num_layers = hme.num_layers
        self.pool_type = hme.pool_type
        self.hidden_dims = hme.hidden_dims[::-1]
        self.module_dict = torch.ModuleDict()
        for i in range(self.num_layers):
            self.module_dict[f'gcn_{i}'] = GCUTPL(in_channels=hme.gcns[self.num_layers - i - 1].out_channels,out_channels=hme.gcns[self.num_layers - i - 1].in_channels, aggr='max')
            if i != self.num_layers - 1:
                self.module_dict[f'pool_{i}'] = HierPool(type=self.pool_type)
        self.module_dict['mlp'] = MLP(self.hidden_dims[0], self.hidden_dims[0])
        self.verbose = hme.verbose

    def forward(self,x0,tpl_edge_indices,As,A_invs,batch,with_layer_xs=False):
        x = self.module_dict['mlp'](x0)
        layer_xs = []
        for i in range(self.num_layers):
            x = ReLU(x)
            x = self.module_dict[f'gcn_{i}'](x, tpl_edge_indices[i])
            layer_xs.append(x)
            if i != self.num_layers - 1:
                x = self.module_dict[f'pool_{i}'](x, As[i])
        if not with_layer_xs:
            return out_x
        return out_x, layer_xs


class AbsSkinWeightPredictors(torch.nn.Module):
    '''
        This abstract class provide basic interface for hierarchical motion retargeting.
    '''
    def __init__(self,input_dims,num_parts,aggrs=('max',),num_layers=None,**kwargs):
        super(AbsSkinWeightPredictors, self).__init__()
        self.input_dims = []
        self.num_parts = []
        self.aggrs = []
        self.num_layers = num_layers if num_layers is not None else len(self.input_dims)
        # align lengths of attributes with that of input_dims.
        for sattr,attr in zip(['input_dims','num_parts','aggrs'],[input_dims,num_parts,aggrs]):
            if isinstance(attr,int):
                self.__setattr__(sattr,[attr]*self.num_layers)
            elif len(attr) == 1:
                self.__setattr__(sattr,list(attr)*self.num_layers)
            else:
                assert len(attr) == self.num_layers, print(f'unmatched length of attr {attr} for num. of layers {self.num_layers}')
                self.__setattr__(sattr,list(attr))
        assert len(self.input_dims) == len(self.num_parts) == len(self.aggrs) == self.num_layers
        # self.modules = torch.nn.ModuleList()
        # for i in range(self.num_layers):
        #     self.modules.append(self._preload_module(layer_id=i))
        ## to save GPU mem., we will not explicitly save all modules since some of them may possess shared params or sub-modules.
        self.module_dict = torch.nn.ModuleDict()
        self.module_dict['shared_modules'] = torch.nn.ModuleDict()
        for layer_id in range(self.num_layers):
            layer_modules,shared_module_dict = self._preload_module(layer_id=layer_id)
            if layer_modules is not None:
                self.module_dict[f'layer_{layer_id}'] = layer_modules
            if shared_module_dict is not None:
                for k in shared_module_dict:
                    if k not in self.module_dict['shared_modules']:
                        self.module_dict['shared_modules'][k] = shared_module_dict[k]

    def _preload_module(self,layer_id):
        raise NotImplementedError

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        # return self.modules[idx]
        raise NotImplementedError # TODO: impl. by return a closure functioned by __forward__.

    def forward_module(self,layer_id, **kwargs):
        raise NotImplementedError

    @property
    def modules(self):
        return self.module_dict

    @property
    def shares(self):
        return self.module_dict['shared_modules']

    def layer_modules(self,layer_id):
        '''
            useful when training/eval/cuda/parameters...()
        '''
        ret_modules = torch.nn.ModuleDict()
        if f'layer_{layer_id}' in self.module_dict:
            ret_modules[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}']
        if f'shared_modules' in self.module_dict:
            ret_modules[f'layer_shared'] = self.module_dict['shared_modules']
        return ret_modules

    def layer_state_dict(self,layer_id,with_shared=False):
        state_dict = {}
        if f'layer_{layer_id}' in self.module_dict:
            state_dict[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}'].state_dict()
        if with_shared:
            state_dict['shared_modules'] = self.shares.state_dict()
        return state_dict

    def load_layer_state_dict(self,layer_id,ckpt,with_shared=False):
        if f'layer_{layer_id}' in self.module_dict:
            self.module_dict[f'layer_{layer_id}'].load_state_dict(ckpt[f'layer_{layer_id}'])
        if with_shared:
            self.module_dict['shared_modules'].load_state_dict(ckpt['shared_modules'])

    def state_dict(self):
        state_dict = {}
        for k in self.module_dict:
            state_dict[k] = self.module_dict[k].state_dict()
        return state_dict

    def load_state_dict(self,ckpt):
        for k in ckpt:
            self.module_dict[k].load_state_dict(ckpt[k])


class SharedSkinWeightPredictors(AbsSkinWeightPredictors):
    '''
        shared SW predictors use shared params. for sw prediction within each hier layer, which consumes the least GPU mem.
        but might be much more difficult to capture details in local texture.
    '''
    def __init__(self,input_dim,num_part,aggr='max',num_layers=None,use_pool=None,**kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim,list) else input_dim
        num_part = num_part[0] if isinstance(num_part,list) else num_part
        aggr = aggr[0] if isinstance(aggr,list) else aggr
        super(SharedSkinWeightPredictors, self).__init__(input_dims=[input_dim],num_parts=[num_part],aggrs=[aggr],num_layers=num_layers,**kwargs)
        self.use_pool = use_pool # could be chosen from None - no inner pool module, 'indp' - inpendent pool modules, 'hier' - hierarchical pool defined by mesh contraction.

    def _preload_module(self,layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        if layer_id > 0:
            return None,None
        shared_module_dict = {}
        shared_module_dict['gcu_1'] = GCUTPL(in_channels=self.input_dims[layer_id], out_channels=64, aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_2'] = GCUTPL(in_channels=64, out_channels=128, aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_3'] = GCUTPL(in_channels=128, out_channels=256, aggr=self.aggrs[layer_id])
        shared_module_dict['mlp_1'] = MLP([(64 + 128 + 256), 256])
        shared_module_dict['mlp_2'] = MLP([256, 256, 128])
        shared_module_dict['mlp_3'] = Linear(128, self.num_parts[layer_id])
        return None,shared_module_dict

    def forward_module(self,layer_id,**kwargs):
        x, tpl_edge_index, batch = kwargs['x'],kwargs['tpl_edge_index'],kwargs['batch']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        pos = x[:, :3]  # 顶点坐标列表
        x_1 = self.shares['gcu_1'](x, tpl_edge_index)  # 对位置做3层graph conv
        x_2 = self.shares['gcu_2'](x_1, tpl_edge_index)
        x_3 = self.shares['gcu_3'](x_2, tpl_edge_index)
        x_4 = self.shares['mlp_1'](torch.cat([x_1, x_2, x_3], dim=1))  # 将所有层叠加，过三层MLP
        x = self.shares['mlp_2'](x_4)
        x = self.shares['mlp_3'](x)
        # softmax
        skinning_weights = torch.softmax(x, 1)  # N vertex assigned to K transformation parts.
        score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0),# sum将所有权重向量在每个图上求和
                                                           torch.bincount(batch),dim=0)  # bincount将求和结果插值扩充成顶点的维度上（按每个图顶点数量）
        weighted_pos = score[:, :, None] * pos[:, None]  # score: [N,K,1] pos [N,1,3]
        weighted_pos = scatter_sum(weighted_pos, batch, dim=0)  # [B,K,3]

        return score, weighted_pos, x, skinning_weights
        # 输出score：[N,K],是按列的归一化（sum N = 1）
        # weighted_pos：[B,K,3],表示每个part的位置
        # x：[N,K]输出的原始分配matrix
        # skinning_weight: [N,K]：是对行的归一化（sum K = 1）

class FusingSkinWeightPredictors(AbsSkinWeightPredictors):
    '''
        Fusing SW predictors use independent params. retargeting on Higher Mesh requires deformation from all the Lower's.
    '''
    def __init__(self,input_dim,aggr='max',num_layers=None,use_pool=None,**kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim,list) else input_dim
        aggr = aggr[0] if isinstance(aggr,list) else aggr
        super(FusingSkinWeightPredictors, self).__init__(input_dims=[input_dim],aggrs=[aggr],num_layers=num_layers,**kwargs)
        self.use_pool = use_pool # could be chosen from None - no inner pool module, 'indp' - inpendent pool modules, 'hier' - hierarchical pool defined by mesh contraction.

    def _preload_module(self,layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        private_module_dict = torch.nn.ModuleDict()
        private_module_dict['gcu_1'] = GCUTPL(in_channels=self.input_dims[layer_id], out_channels=64, aggr=self.aggrs[layer_id])
        private_module_dict['gcu_2'] = GCUTPL(in_channels=64, out_channels=128, aggr=self.aggrs[layer_id])
        private_module_dict['gcu_3'] = GCUTPL(in_channels=128, out_channels=256, aggr=self.aggrs[layer_id])
        private_module_dict['mlp_1'] = MLP([(64 + 128 + 256), 256])
        private_module_dict['mlp_2'] = MLP([256, 256, 128])
        private_module_dict['mlp_3'] = Linear(128, self.num_parts[layer_id])
        return private_module_dict,None

    def forward_module(self,layer_id,**kwargs):
        x, tpl_edge_index, batch = kwargs['x'],kwargs['tpl_edge_index'],kwargs['batch']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        pos = x[:, :3]  # 顶点坐标列表
        cur_module = self.modules[f'layer_{layer_id}']
        x_1 = cur_module['gcu_1'](x, tpl_edge_index)  # 对位置做3层graph conv
        x_2 = cur_module['gcu_2'](x_1, tpl_edge_index)
        x_3 = cur_module['gcu_3'](x_2, tpl_edge_index)
        x_4 = cur_module['mlp_1'](torch.cat([x_1, x_2, x_3], dim=1))  # 将所有层叠加，过三层MLP
        x = cur_module['mlp_2'](x_4)
        x = cur_module['mlp_3'](x)
        # softmax
        skinning_weights = torch.softmax(x, 1)  # N vertex assigned to K transformation parts.
        score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0),# sum将所有权重向量在每个图上求和
                                                           torch.bincount(batch),dim=0)  # bincount将求和结果插值扩充成顶点的维度上（按每个图顶点数量）
        weighted_pos = score[:, :, None] * pos[:, None]  # score: [N,K,1] pos [N,1,3]
        weighted_pos = scatter_sum(weighted_pos, batch, dim=0)  # [B,K,3]

        return score, weighted_pos, x, skinning_weights
        # 输出score：[N,K],是按列的归一化（sum N = 1）
        # weighted_pos：[B,K,3],表示每个part的位置
        # x：[N,K]输出的原始分配matrix
        # skinning_weight: [N,K]：是对行的归一化（sum K = 1）

class HierSkinWeightPredictors(AbsSkinWeightPredictors):
    '''
        hierarchical SW Predictor leverage definite graph pooling to learn extra high-freq info for each LR Mesh.
    '''
    def __init__(self,input_dim,num_part,aggr='max',num_layers=None,use_pool=None,**kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim,list) else input_dim
        num_part = num_part[0] if isinstance(num_part,list) else num_part
        aggr = aggr[0] if isinstance(aggr,list) else aggr
        super(HierSkinWeightPredictors, self).__init__(input_dims=[input_dim],num_parts=[num_part],aggrs=[aggr],num_layers=num_layers,**kwargs)
        self.use_pool = use_pool # could be chosen from None - no inner pool module, 'indp' - inpendent pool modules, 'hier' - hierarchical pool defined by mesh contraction.

    def _preload_module(self,layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        shared_module_dict = {}
        shared_module_dict['gcu_1'] = GCUTPL(in_channels=self.input_dims[layer_id], out_channels=64, aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_2'] = GCUTPL(in_channels=64, out_channels=128, aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_3'] = GCUTPL(in_channels=128, out_channels=256, aggr=self.aggrs[layer_id])
        shared_module_dict['mlp_1'] = MLP([(64 + 128 + 256), 256])
        shared_module_dict['mlp_2'] = MLP([256, 256, 128])
        shared_module_dict['mlp_3'] = Linear(128, self.num_parts[layer_id])
        return None,shared_module_dict

    def forward_module(self,layer_id,**kwargs):
        x, tpl_edge_index, batch = kwargs['x'],kwargs['tpl_edge_index'],kwargs['batch']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        pos = x[:, :3]  # 顶点坐标列表
        x_1 = self.shares['gcu_1'](x, tpl_edge_index)  # 对位置做3层graph conv
        x_2 = self.shares['gcu_2'](x_1, tpl_edge_index)
        x_3 = self.shares['gcu_3'](x_2, tpl_edge_index)
        x_4 = self.shares['mlp_1'](torch.cat([x_1, x_2, x_3], dim=1))  # 将所有层叠加，过三层MLP
        x = self.shares['mlp_2'](x_4)
        x = self.shares['mlp_3'](x)
        # softmax
        skinning_weights = torch.softmax(x, 1)  # N vertex assigned to K transformation parts.
        score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0),# sum将所有权重向量在每个图上求和
                                                           torch.bincount(batch),dim=0)  # bincount将求和结果插值扩充成顶点的维度上（按每个图顶点数量）
        weighted_pos = score[:, :, None] * pos[:, None]  # score: [N,K,1] pos [N,1,3]
        weighted_pos = scatter_sum(weighted_pos, batch, dim=0)  # [B,K,3]

        return score, weighted_pos, x, skinning_weights
        # 输出score：[N,K],是按列的归一化（sum N = 1）
        # weighted_pos：[B,K,3],表示每个part的位置
        # x：[N,K]输出的原始分配matrix
        # skinning_weight: [N,K]：是对行的归一化（sum K = 1）


class AbsPartEncoders(torch.nn.Module):
    '''
        This abstract class provide basic interface for hierarchical motion retargeting.
    '''

    def __init__(self, input_dims, output_dims, aggrs=('max',), num_layers=None, **kwargs):
        super(AbsPartEncoders, self).__init__()
        self.input_dims = []
        self.output_dims = []
        self.aggrs = []
        self.num_layers = num_layers if num_layers is not None else len(self.input_dims)
        # align lengths of attributes with that of input_dims.
        for sattr, attr in zip(['input_dims', 'output_dims', 'aggrs'], [input_dims, output_dims, aggrs]):
            if isinstance(attr,int):
                self.__setattr__(sattr,[attr]*self.num_layers)
            elif len(attr) == 1:
                self.__setattr__(sattr, list(attr) * self.num_layers)
            else:
                assert len(attr) == self.num_layers, print(
                    f'unmatched length of attr {attr} for num. of layers {self.num_layers}')
                self.__setattr__(sattr, list(attr))
        assert len(self.input_dims) == len(self.output_dims) == len(self.aggrs) == self.num_layers
        # self.modules = torch.nn.ModuleList()
        # for i in range(self.num_layers):
        #     self.modules.append(self._preload_module(layer_id=i))
        ## to save GPU mem., we will not explicitly save all modules since some of them may possess shared params or sub-modules.
        self.module_dict = torch.nn.ModuleDict()
        self.module_dict['shared_modules'] = torch.nn.ModuleDict()
        for layer_id in range(self.num_layers):
            layer_modules, shared_module_dict = self._preload_module(layer_id=layer_id)
            if layer_modules is not None:
                self.module_dict[f'layer_{layer_id}'] = layer_modules
            if shared_module_dict is not None:
                for k in shared_module_dict:
                    if k not in self.module_dict['shared_modules']:
                        self.module_dict['shared_modules'][k] = shared_module_dict[k]

    def _preload_module(self, layer_id):
        raise NotImplementedError

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        # return self.modules[idx]
        raise NotImplementedError  # TODO: impl. by return a closure functioned by __forward__.

    def forward_module(self, layer_id, **kwargs):
        raise NotImplementedError

    @property
    def modules(self):
        return self.module_dict

    @property
    def shares(self):
        return self.module_dict['shared_modules']

    def layer_modules(self,layer_id):
        '''
            useful when training/eval/cuda/parameters...()
        '''
        ret_modules = torch.nn.ModuleDict()
        if f'layer_{layer_id}' in self.module_dict:
            ret_modules[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}']
        if f'shared_modules' in self.module_dict:
            ret_modules[f'layer_shared'] = self.module_dict['shared_modules']
        return ret_modules

    def layer_state_dict(self, layer_id, with_shared=False):
        state_dict = {}
        if f'layer_{layer_id}' in self.module_dict:
            state_dict[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}'].state_dict()
        if with_shared:
            state_dict['shared_modules'] = self.shares.state_dict()
        return state_dict

    def load_layer_state_dict(self, layer_id, ckpt, with_shared=False):
        if f'layer_{layer_id}' in self.module_dict:
            self.module_dict[f'layer_{layer_id}'].load_state_dict(ckpt[f'layer_{layer_id}'])
        if with_shared:
            self.module_dict['shared_modules'].load_state_dict(ckpt['shared_modules'])

    def state_dict(self):
        state_dict = {}
        for k in self.module_dict:
            state_dict[k] = self.module_dict[k].state_dict()
        return state_dict

    def load_state_dict(self, ckpt):
        for k in ckpt:
            self.module_dict[k].load_state_dict(ckpt[k])

class FusingPartEncoders(AbsPartEncoders):
    '''
        shared part encoders use shared params. for encoding within each hier layer, which consumes the least GPU mem.
        but might be much more difficult to capture details in local texture.
    '''

    def __init__(self, input_dim, aggr='max', num_layers=None, use_pool=None, **kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim,list) else input_dim
        # output_dim = output_dim[0] if isinstance(output_dim,list) else output_dim
        aggr = aggr[0] if isinstance(aggr,list) else aggr
        super(FusingPartEncoders, self).__init__(input_dims=[input_dim], aggrs=[aggr],
                                                         num_layers=num_layers, **kwargs)
        self.use_pool = use_pool  # could be chosen from None - no inner pool module, 'indp' - inpendent pool modules, 'hier' - hierarchical pool defined by mesh contraction.
        # TODO: evalute if there is any improvement when interaction among parts are explicitly (or both along with implicit) for posed encoding.

    def _preload_module(self, layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        private_module_dict = torch.nn.ModuleDict()
        private_module_dict['gcu_1'] = GCUTPL(in_channels=self.input_dims[layer_id], out_channels=64,
                                           aggr=self.aggrs[layer_id])
        private_module_dict['gcu_2'] = GCUTPL(in_channels=64, out_channels=128, aggr=self.aggrs[layer_id])
        private_module_dict['gcu_3'] = GCUTPL(in_channels=128, out_channels=256, aggr=self.aggrs[layer_id])

        private_module_dict['mlp_1'] = MLP([(64 + 128 + 256), 256])
        private_module_dict['mlp_2'] = MLP([(64 + 128 + 256), 256])  # g-conv和mlp_glb部分的参数配置和predictor完全相同

        private_module_dict['lin'] = Linear(512, self.output_dims[layer_id])
        private_module_dict['relu'] = ReLU()
        private_module_dict['norm'] = BatchNorm1d(self.output_dims[layer_id], momentum=0.1)

        return private_module_dict,None

    def forward_module(self, layer_id, **kwargs):
        pos, hm, tpl_edge_index, batch, feat = kwargs['pos'], kwargs['hm'], kwargs['tpl_edge_index'], kwargs['batch'], kwargs['feat']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')

        cur_module = self.modules[f'layer_{layer_id}']

        x_in = torch.cat((pos, feat), 1) # 输入x包含位置和法向量
        x_1 = cur_module['gcu_1'](x_in, tpl_edge_index)
        x_2 = cur_module['gcu_2'](x_1, tpl_edge_index)
        x_3 = cur_module['gcu_3'](x_2, tpl_edge_index)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        x_4 = cur_module['mlp_1'](x_123)  # (B*V, 256)
        x_global, _ = scatter_max(x_123, batch, dim=0)  # (B, C) 每个batch上计算特征的最大值
        x_global = cur_module['mlp_2'](x_global) # embed最大值信息
        x_global = torch.repeat_interleave(x_global, torch.bincount(batch), dim=0)  # (B*V, 256) 将最大值扩展到batch的所有顶点维度

        x_5 = torch.cat((x_4, x_global), 1) # (B*V, 512) 将最大值信息和每个节点concat继续计算
        x_6 = scatter_sum(x_5[:, None] * hm[:, :, None], batch, dim=0)  # (B, K, 512)，其中，乘积得到[N,K,2C]，hm的作用是把V上的特征按V归一化地转移到part上
        x_6 = cur_module['lin'](x_6)
        x_6 = cur_module['relu'](x_6)
        x_6 = x_6.permute(0, 2, 1) # 为了对数据维度归一化，而不是part维度
        y = cur_module['norm'](x_6).permute(0, 2, 1)
        return y

class SharedPartEncoders(AbsPartEncoders):
    '''
        shared part encoders use shared params. for encoding within each hier layer, which consumes the least GPU mem.
        but might be much more difficult to capture details in local texture.
    '''

    def __init__(self, input_dim, output_dim, aggr='max', num_layers=None, use_pool=None, **kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim,list) else input_dim
        output_dim = output_dim[0] if isinstance(output_dim,list) else output_dim
        aggr = aggr[0] if isinstance(aggr,list) else aggr
        super(SharedPartEncoders, self).__init__(input_dims=[input_dim], output_dims=[output_dim], aggrs=[aggr],
                                                         num_layers=num_layers, **kwargs)
        self.use_pool = use_pool  # could be chosen from None - no inner pool module, 'indp' - inpendent pool modules, 'hier' - hierarchical pool defined by mesh contraction.
        # TODO: evalute if there is any improvement when interaction among parts are explicitly (or both along with implicit) for posed encoding.

    def _preload_module(self, layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        if layer_id > 0:
            return None,None
        shared_module_dict = {}
        shared_module_dict['gcu_1'] = GCUTPL(in_channels=self.input_dims[layer_id], out_channels=64,
                                           aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_2'] = GCUTPL(in_channels=64, out_channels=128, aggr=self.aggrs[layer_id])
        shared_module_dict['gcu_3'] = GCUTPL(in_channels=128, out_channels=256, aggr=self.aggrs[layer_id])

        shared_module_dict['mlp_1'] = MLP([(64 + 128 + 256), 256])
        shared_module_dict['mlp_2'] = MLP([(64 + 128 + 256), 256])  # g-conv和mlp_glb部分的参数配置和predictor完全相同

        shared_module_dict['lin'] = Linear(512, self.output_dims[layer_id])
        shared_module_dict['relu'] = ReLU()
        shared_module_dict['norm'] = BatchNorm1d(self.output_dims[layer_id], momentum=0.1)

        return None,shared_module_dict

    def forward_module(self, layer_id, **kwargs):
        pos, hm, tpl_edge_index, batch, feat = kwargs['pos'], kwargs['hm'], kwargs['tpl_edge_index'], kwargs['batch'], kwargs['feat']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        x_in = torch.cat((pos, feat), 1) # 输入x包含位置和法向量
        x_1 = self.shares['gcu_1'](x_in, tpl_edge_index)
        x_2 = self.shares['gcu_2'](x_1, tpl_edge_index)
        x_3 = self.shares['gcu_3'](x_2, tpl_edge_index)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        x_4 = self.shares['mlp_1'](x_123)  # (B*V, 256)
        x_global, _ = scatter_max(x_123, batch, dim=0)  # (B, C) 每个batch上计算特征的最大值
        x_global = self.shares['mlp_2'](x_global) # embed最大值信息
        x_global = torch.repeat_interleave(x_global, torch.bincount(batch), dim=0)  # (B*V, 256) 将最大值扩展到batch的所有顶点维度

        x_5 = torch.cat((x_4, x_global), 1) # (B*V, 512) 将最大值信息和每个节点concat继续计算
        x_6 = scatter_sum(x_5[:, None] * hm[:, :, None], batch, dim=0)  # (B, K, 512)，其中，乘积得到[N,K,2C]，hm的作用是把V上的特征按V归一化地转移到part上
        x_6 = self.shares['lin'](x_6)
        x_6 = self.shares['relu'](x_6)
        x_6 = x_6.permute(0, 2, 1) # 为了对数据维度归一化，而不是part维度
        y = self.shares['norm'](x_6).permute(0, 2, 1)
        return y
class AbsPartDecoders(torch.nn.Module):
    '''
        This abstract class provide basic interface for hierarchical motion retargeting.
    '''

    def __init__(self, input_dims, num_layers=None, **kwargs):
        super(AbsPartDecoders, self).__init__()
        self.input_dims = []
        self.num_layers = num_layers if num_layers is not None else len(self.input_dims)
        # align lengths of attributes with that of input_dims.
        for sattr, attr in zip(['input_dims'], [input_dims]):
            if isinstance(attr,int):
                self.__setattr__(sattr,[attr]*self.num_layers)
            elif len(attr) == 1:
                self.__setattr__(sattr, list(attr) * self.num_layers)
            else:
                assert len(attr) == self.num_layers, print(
                    f'unmatched length of attr {attr} for num. of layers {self.num_layers}')
                self.__setattr__(sattr, list(attr))
        assert len(self.input_dims) == self.num_layers
        self.module_dict = torch.nn.ModuleDict()
        self.module_dict['shared_modules'] = torch.nn.ModuleDict()
        for layer_id in range(self.num_layers):
            layer_modules, shared_module_dict = self._preload_module(layer_id=layer_id)
            if layer_modules is not None:
                self.module_dict[f'layer_{layer_id}'] = layer_modules
            if shared_module_dict is not None:
                for k in shared_module_dict:
                    if k not in self.module_dict['shared_modules']:
                        self.module_dict['shared_modules'][k] = shared_module_dict[k]

    def _preload_module(self, layer_id):
        raise NotImplementedError

    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx):
        # return self.modules[idx]
        raise NotImplementedError  # TODO: impl. by return a closure functioned by __forward__.

    def forward_module(self, layer_id, **kwargs):
        raise NotImplementedError

    @property
    def modules(self):
        return self.module_dict

    @property
    def shares(self):
        return self.module_dict['shared_modules']

    def layer_modules(self, layer_id):
        '''
            useful when training/eval/cuda/parameters...()
        '''
        ret_modules = torch.nn.ModuleDict()
        if f'layer_{layer_id}' in self.module_dict:
            ret_modules[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}']
        if f'shared_modules' in self.module_dict:
            ret_modules[f'layer_shared'] = self.module_dict['shared_modules']
        return ret_modules

    def layer_state_dict(self, layer_id, with_shared=False):
        state_dict = {}
        if f'layer_{layer_id}' in self.module_dict:
            state_dict[f'layer_{layer_id}'] = self.module_dict[f'layer_{layer_id}'].state_dict()
        if with_shared:
            state_dict['shared_modules'] = self.shares.state_dict()
        return state_dict

    def load_layer_state_dict(self, layer_id, ckpt, with_shared=False):
        if f'layer_{layer_id}' in self.module_dict:
            self.module_dict[f'layer_{layer_id}'].load_state_dict(ckpt[f'layer_{layer_id}'])
        if with_shared:
            self.module_dict['shared_modules'].load_state_dict(ckpt['shared_modules'])

    def state_dict(self):
        state_dict = {}
        for k in self.module_dict:
            state_dict[k] = self.module_dict[k].state_dict()
        return state_dict

    def load_state_dict(self, ckpt):
        for k in ckpt:
            self.module_dict[k].load_state_dict(ckpt[k])

class SharedPartDecoders(AbsPartDecoders):
    '''
        shared part decoders use shared params. for decoding within each hier layer, which consumes the least GPU mem.
        but might be much more difficult to capture details in local texture.
    '''

    def __init__(self, input_dim,num_layer=None,**kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        input_dim = input_dim[0] if isinstance(input_dim, list) else input_dim
        super(SharedPartDecoders, self).__init__(input_dims=[input_dim],num_layers=num_layer, **kwargs)
        # TODO: evalute if there is any improvement when interaction among parts are explicitly (or both along with implicit) for posed encoding.

    def _preload_module(self, layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        if layer_id > 0:
            return None,None
        net = [Linear(self.input_dims[layer_id], 256), ReLU(inplace=True),
               Linear(256, 128), ReLU(inplace=True),
               Linear(128, 128), ReLU(inplace=True),
               Linear(128, 7)]  # 输出3+4维度表示位移和旋转
        shared_module_dict = {}
        shared_module_dict['net'] = Sequential(*net)
        return None,shared_module_dict

    def forward_module(self, layer_id, **kwargs):
        feats = kwargs['feats']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        # feat: (B, K, C)
        x = torch.cat(feats, 2)
        x = self.shares['net'](x)
        if 'base' in kwargs:
            x = x + kwargs['base']
        return x

class FusingPartDecoders(AbsPartDecoders):
    '''
    '''

    def __init__(self,num_layer=None,**kwargs):
        '''
            only fixed hyper-param allowed in this case.
        '''
        # input_dim = input_dim[0] if isinstance(input_dim, list) else input_dim
        super(FusingPartDecoders, self).__init__(num_layers=num_layer, **kwargs)
        # TODO: evalute if there is any improvement when interaction among parts are explicitly (or both along with implicit) for posed encoding.

    def _preload_module(self, layer_id):
        # assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        net = [Linear(self.input_dims[layer_id], 256), ReLU(inplace=True),
               Linear(256, 128), ReLU(inplace=True),
               Linear(128, 128), ReLU(inplace=True),
               Linear(128, 7)]  # 输出3+4维度表示位移和旋转
        private_module_dict = torch.nn.ModuleDict()
        private_module_dict['net'] = Sequential(*net)
        return private_module_dict,None

    def forward_module(self, layer_id, **kwargs):
        feats = kwargs['feats']
        assert 0 <= layer_id < len(self), print(f'illegal visit for layer id {layer_id} in a {len(self)}-layer module list.')
        # feat: (B, K, C)
        x = torch.cat(feats, 2)
        x = self.modules[f'layer_{layer_id}']['net'](x)
        if 'base' in kwargs:
            x = x + kwargs['base']
        return x