import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from src.utils.lbs import lbs
from src.utils.geometry import get_tpl_edges, fps_np, get_normal
from src.utils.o3d_wrapper import Mesh, MeshO3d


class MotionTestDataset(Dataset):
    def __init__(self, data_dir,default_name='test1',preload=True):
        super(MotionTestDataset, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        self.names = list(os.listdir(self.data_dir))
        self.vs, self.fs = [], None
        self.v0, self.normal_v0 = None, None
        self.tpl_edge_indexs = None
        self.cur_name = None
        if preload:
            try:
                self.set_motion(default_name)
            except:
                print('fail to preload custom dataset.')

        print('Number of poses:', len(self))

    def set_motion(self,name):
        assert name in self.names
        if self.cur_name == name:
            return
        self.cur_name = name
        self._preload()


    def get(self, index):
        v, _, _, name = self.load(index)
        v1 = (v - self.center) / self.scale
        v1 = torch.from_numpy(v1).float()
        normal_v1 = get_normal(v1, self.f)
        return Data(v0=self.v0, v1=v1, tpl_edge_index=self.tpl_edge_index, triangle=self.f[None].astype(int),
                    feat0=self.normal_v0, feat1=normal_v1,
                    name=name, num_nodes=len(v1))

    def get_tpose_only(self):
        return Data(v0=self.v0, v1=None, tpl_edge_index=self.tpl_edge_index, triangle=self.f[None].astype(int),
                    feat0=self.normal_v0, feat1=None,
                    name='tpose', num_nodes=len(self.v0))

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            # return self.vs[index], self.fs, self.tpl_edge_indexs, self.names[index]
            return self.vs[index], self.fs, self.tpl_edge_indexs, f'{self.cur_name}_frm_{index}'

    def len(self):
        # return len(self.names)
        return self.num_pose

    def _preload(self):
        # rest mesh
        mesh_path = os.path.join(self.data_dir,self.cur_name, f'{self.cur_name}-tpose.obj')
        m = Mesh(filename=mesh_path)
        self.v0 = m.v
        self.f = m.f
        tpl_edge_index = get_tpl_edges(m.v, m.f)
        tpl_edge_index = tpl_edge_index.astype(int).T
        tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
        self.tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=self.v0.shape[0])

        self.center = (np.max(self.v0, 0, keepdims=True) + np.min(self.v0, 0, keepdims=True)) / 2
        self.scale = np.max(self.v0[:, 1], 0) - np.min(self.v0[:, 1], 0)
        self.v0 = (self.v0 - self.center) / self.scale
        self.v0 = torch.from_numpy(self.v0).float()
        self.normal_v0 = get_normal(self.v0, self.f)

        # posed mesh
        list_files = list(os.listdir(os.path.join(self.data_dir, self.cur_name)))
        filtered_files = []
        for file in list_files:
            if file.endswith('.obj'):
                filtered_files.append(file)
        self.num_pose = len(filtered_files) - 1 # without t-pose.
        self.vs = []
        if self.num_pose > 0:
            for idx in range(self.num_pose):
                mesh_path = os.path.join(self.data_dir,self.cur_name,f'{self.cur_name}-{idx+1}.obj')
                m = Mesh(filename=mesh_path)
                self.vs.append(m.v)
        else:
            # has no pose.
            pass


