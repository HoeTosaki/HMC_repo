import pandas as pd
from time import time
import random
import torch as tc
from torch_geometric.loader import DataLoader
import tqdm
import os

from src.utils import *
from src.utils.o3d_wrapper import Mesh
from src.data_utils.custom_loader import MotionTestDataset
from src.training.preq import prepare_modules
from src.hier_models.ops import *
from src.utils.visualization import *
from src.utils.geometry import get_normal,get_tpl_edges_mod
from src.hier_models.hier_tools import HierMesh
from torch_geometric.data import Dataset, Data


# init env.
torch.multiprocessing.set_sharing_strategy('file_system')
np.seterr(divide='ignore',invalid='ignore')
glb_st:Config = Config(name='global_storage')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# init model cfg.
from src.config.config_repo import fusing_frate_64_var_config as model_cfg

# def. of HMC model for inference.
class HMC:
    def __init__(self,cfg:Config,model_path,verbose=False):
        self.layer_id = None
        self.cfg:Config = cfg
        self.model_path:str = model_path
        self.verbose:bool = verbose

    def _hier_eval_closure(self,module_type='pred'):
        '''
            adaptor between hier modules & basic modules.
        '''
        def pred_closure(x,data,verbose=True,layer_id=None):
            return self.predictor.forward_module(layer_id=self.layer_id if layer_id is None else layer_id,x=x,batch=data.batch,tpl_edge_index=data.tpl_edge_index)

        def enc_closure(v0,hm,data,feat,layer_id=None):
            return self.encoder_shape.forward_module(layer_id=self.layer_id if layer_id is None else layer_id, pos=v0, hm=hm, tpl_edge_index=data.tpl_edge_index,batch=data.batch, feat=feat)

        def dec_closure(*args,layer_id=None):
            return self.decoder.forward_module(layer_id=self.layer_id if layer_id is None else layer_id,feats=args)

        if module_type == 'pred':
            return pred_closure
        elif module_type == 'enc':
            return enc_closure
        elif module_type == 'dec':
            return  dec_closure
        else:
            raise NotImplementedError

    @classmethod
    def coarse_mesh_data(cls, mesh_data, pooling_ratio, with_hm=False, cache_hm=None, num_layers=2, output_layer_id=1):
        '''
            input an embedded data of mesh, turn it into coarse mesh data.
            the result data is only available for inference, since the sw / joint pos remain agnostic,
             forbidding div_sw_loss & augmentation.
        '''
        if cache_hm is None:
            hm = HierMesh(v=mesh_data.v0.cpu().detach().numpy(), f=mesh_data.triangle[0], pooling_ratio=pooling_ratio,
                          num_layers=num_layers, pooling_type='face')
            hm.load_all_meshes(assign_preferred=False)
            hm.compute_all_assign(is_bary_saved=False, has_inv_A=True)
        else:
            hm = cache_hm
        if output_layer_id == 0:
            if with_hm:
                return mesh_data, hm
            else:
                return mesh_data

        if num_layers == 2:
            # only one coarse mesh exist for this case.
            # construct new Data with coarasened mesh.
            dw_mesh = Data(name='visual-tmp', num_nodes=hm.meshes[1][0].shape[0])
            # compute new v0, feat0, etc.
            dw_mesh.v0 = tc.from_numpy(hm.meshes[1][0]).float()
            dw_mesh.triangle = hm.meshes[1][1][None]
            dw_mesh.feat0 = get_normal(dw_mesh.v0, hm.meshes[1][1])
            dw_mesh.tpl_edge_index = HierMesh.read_mesh_edge(hm.meshes[1])
            # compute new v1, feat1.
            if hasattr(mesh_data,'v1') and mesh_data.v1 is not None:
                dw_mesh.v1 = tc.from_numpy(
                    np.matmul(hm.save_param_dicts[0]['A_inv'].A, mesh_data.v1.cpu().detach().numpy())).float()
                dw_mesh.feat1 = get_normal(dw_mesh.v1, hm.meshes[1][1])
        else:
            # construct new Data with coarasened mesh.
            dw_mesh = Data(name=f'visual-tmp-l{output_layer_id}', num_nodes=hm.meshes[output_layer_id][0].shape[0])
            # compute new v0, feat0, etc.
            dw_mesh.v0 = tc.from_numpy(hm.meshes[output_layer_id][0]).float()
            dw_mesh.triangle = hm.meshes[output_layer_id][1][None]
            dw_mesh.feat0 = get_normal(dw_mesh.v0, hm.meshes[output_layer_id][1])
            dw_mesh.tpl_edge_index = HierMesh.read_mesh_edge(hm.meshes[output_layer_id])
            # compute new v1, feat1.
            mul_times = output_layer_id
            cur_v1 = mesh_data.v1.cpu().detach().numpy()
            for cur_time in range(mul_times):
                cur_v1 = np.matmul(hm.save_param_dicts[cur_time]['A_inv'].A, cur_v1)
            dw_mesh.v1 = tc.from_numpy(cur_v1).float()
            dw_mesh.feat1 = get_normal(dw_mesh.v1, hm.meshes[output_layer_id][1])
        if with_hm:
            return dw_mesh, hm
        return dw_mesh

    def load_model(self,device=None):
        ckpt = tc.load(self.model_path)
        # optimizer.load_state_dict(ckpt['optimizer'])
        if self.verbose:
            model_names = self.model_path.split("/")
            print(f'load model: name={"-".join(model_names[-2:]) if len(model_names) >= 2 else "-".join(model_names)} | end_epoch={ckpt["epoch"]}')

        self.predictor,self.encoder_shape,self.decoder = prepare_modules(self.cfg)
        self.predictor.load_state_dict(ckpt=ckpt['pred'])
        self.encoder_shape.load_state_dict(ckpt=ckpt['enc'])
        self.decoder.load_state_dict(ckpt=ckpt['dec'])

        if device is not None:
            self.predictor.to(device)
            self.encoder_shape.to(device)
            self.decoder.to(device)
        # pack closure for each module.
        self.pred_cls = self._hier_eval_closure(module_type='pred')
        self.enc_cls = self._hier_eval_closure(module_type='enc')
        self.dec_cls = self._hier_eval_closure(module_type='dec')

    def hier_retarget_pose(self, src_data, dst_data, src_hm=None, dst_hm=None, with_inner_timer=False,precoarsen_src=None):
        assert src_data.batch is None and dst_data.batch is None
        pooling_ratio = self.cfg.pooling_ratio
        self.predictor.eval()
        self.encoder_shape.eval()
        self.decoder.eval()
        fusing_rates = self.cfg.layer_final_fusing_rates
        iter_fusing_rates = rect_iter_rates(fusing_rates)

        hd_pos_prev = None
        region_score_prev = None
        pred_disp_prev = None
        pre_src_hm = None

        if precoarsen_src is not None:
            src_hm,pre_src_hm = src_hm if (isinstance(src_hm, tuple) or isinstance(src_hm,list)) else (None,None)
            src_data,pre_src_hm = self.coarse_mesh_data(src_data,pooling_ratio=precoarsen_src,cache_hm=pre_src_hm,with_hm=True,num_layers=2,output_layer_id=1)

        run_time_per_layers = [0] * self.cfg.num_layers
        retargeted_meshes = []
        for layer_id in reversed(range(self.cfg.num_layers)):
            # TODO & Caution: the following matmul will cost a bad time complexity of O(N^2), since each layer with all previous layers.
            # prepare data.
            run_time_per_layers[layer_id] = time()
            src_dw_data, src_hm = self.coarse_mesh_data(src_data, pooling_ratio=pooling_ratio, with_hm=True,cache_hm=src_hm,num_layers=self.cfg.num_layers,output_layer_id=layer_id)
            dst_dw_data, dst_hm = self.coarse_mesh_data(dst_data, pooling_ratio=pooling_ratio, with_hm=True,cache_hm=dst_hm,num_layers=self.cfg.num_layers,output_layer_id=layer_id)

            src_dw_data = next(DataLoader([src_dw_data], batch_size=1, shuffle=False, pin_memory=False,drop_last=False)._get_iterator())
            dst_dw_data = next(DataLoader([dst_dw_data], batch_size=1, shuffle=False, pin_memory=False,drop_last=False)._get_iterator())

            # inference by current layer.
            hm0, hd0, _, region_score0 = self.pred_cls(torch.cat((dst_dw_data.v0, dst_dw_data.feat0), 1), data=dst_dw_data,verbose=True,layer_id=layer_id)
            hm1, hd1, _, region_score1 = self.pred_cls(torch.cat((src_dw_data.v0, src_dw_data.feat0), 1), data=src_dw_data,verbose=True,layer_id=layer_id)
            trans1 = get_transformation(hm1, region_score1, src_dw_data.batch, src_dw_data.v0, src_dw_data.v1)
            pose_enc_0 = self.enc_cls(src_dw_data.v0, hm1, data=src_dw_data, feat=src_dw_data.feat0,layer_id=layer_id)
            pose_enc = self.enc_cls(src_dw_data.v1, hm1, data=src_dw_data, feat=src_dw_data.feat1,layer_id=layer_id)
            shape_enc = self.enc_cls(dst_dw_data.v0, hm0, data=dst_dw_data, feat=dst_dw_data.feat0,layer_id=layer_id)
            pred_disp = self.dec_cls(pose_enc - pose_enc_0, shape_enc, trans1,layer_id=layer_id) # pred disp from the current layer.

            if layer_id != self.cfg.num_layers - 1:
                # concat the previous retargeting results from the LR mesh.
                hd0_comb = tc.cat([hd0, hd_pos_prev], dim=-2)
                _region_score_prev = tc.matmul(tc.from_numpy(dst_hm.save_param_dicts[layer_id]['A'].A).float(),region_score_prev)
                region_score0_comb = tc.cat([region_score0, _region_score_prev], dim=-1)
                pred_disp_comb = tc.cat([pred_disp, pred_disp_prev], dim=-2)
                region_score0_comb[:, :region_score0.shape[-1]] *= 1 - iter_fusing_rates[layer_id]
                region_score0_comb[:, region_score0.shape[-1]:] *= iter_fusing_rates[layer_id]

                pred_v = handle2mesh(pred_disp_comb, hd0_comb, region_score0_comb, dst_dw_data.batch, dst_dw_data.v0)
                v, f, vc = visualize_handle(pred_v.cpu().detach().numpy(), dst_dw_data.triangle[0][0],save_path=None)
                retargeted_meshes.append(Mesh(v=v[0],f=f[0],vc=vc[0]))

                pred_disp_prev = pred_disp_comb
                hd_pos_prev = hd0_comb
                region_score_prev = region_score0_comb
            else:
                pred_v = handle2mesh(pred_disp, hd0, region_score0, dst_dw_data.batch, dst_dw_data.v0)
                v, f, vc = visualize_handle(pred_v.cpu().detach().numpy(), dst_dw_data.triangle[0][0], save_path=None)
                retargeted_meshes.append(Mesh(v=v[0], f=f[0], vc=vc[0]))

                pred_disp_prev = pred_disp
                hd_pos_prev = hd0
                region_score_prev = region_score0

            run_time_per_layers[layer_id] = time() - run_time_per_layers[layer_id]
        if precoarsen_src is not None:
            src_hm = (src_hm,pre_src_hm)
        if with_inner_timer:
            return retargeted_meshes[::-1], run_time_per_layers[::-1], src_hm, dst_hm
        else:
            return retargeted_meshes[::-1], src_hm, dst_hm


def load_model(model_path,device=None):
    global model_cfg
    hmc = HMC(model_cfg,model_path,verbose=False)
    hmc.load_model(device=device)
    glb_st.param = ('hmc',hmc)

def retarget_pose(src_tpose_path,src_pose_path,tgt_tpose_path,out_tgt_pose_path):
    raise NotImplementedError

def retarget_motion(src_name,tgt_name,precoarsen_src=None):
    '''
        src_dir contains {src_name}-tpose.obj and {src_name}-%d.obj (int from 1 to XXXX)
        tgt_dir contains {tgt_name}-tpose.obj
    '''
    src_dataset = MotionTestDataset(data_dir='./data',default_name=src_name)
    tgt_dataset = MotionTestDataset(data_dir='./data', default_name=tgt_name)
    hmc:HMC = glb_st.hmc

    dst_data = tgt_dataset.get_tpose_only()

    src_hm,dst_hm = None, None
    for src_idx,src_data in enumerate(tqdm.tqdm(src_dataset)):
        ret_meshes,src_hm,dst_hm = hmc.hier_retarget_pose(src_data=src_data,dst_data=dst_data,src_hm=src_hm,dst_hm=dst_hm,with_inner_timer=False,precoarsen_src=precoarsen_src)
        ret_meshes[0].write_obj(os.path.join('./data',tgt_name,f'{tgt_name}-{src_idx:06d}.obj'))




if __name__ == '__main__':
    load_model('./pretrained/pretrained_hmc.ckpt',device=tc.device('cpu'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_name',
        default='greeting',
        type=str,
        help='name of source motion dir.'
    )
    parser.add_argument(
        '--tgt_name',
        default='greeting_on_target',
        type=str,
        help='name of target character dir'
    )
    parser.add_argument(
        '--precoarsen_src',
        default=0.4,
        type=float,
        help='whether to use pre-coarsening on input motion to accelerate inference. set it to 0 if is forbidden.'
    )
    args = parser.parse_args()
    retarget_motion(src_name=args.src_name,tgt_name=args.tgt_name,precoarsen_src=args.precoarsen_src if args.precoarsen_src != 0 else None)
