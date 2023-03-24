from src.utils import *
import cv2
import torch
from torch_geometric.loader import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from src.hier_models.networks import SharedSkinWeightPredictors,SharedPartEncoders,SharedPartDecoders
from src.hier_models.networks import FusingSkinWeightPredictors,FusingPartEncoders, FusingPartDecoders
from src.hier_models.smpl import SMPL2Mesh
from src.hier_models.networks import HierPool,HierMeshDecoder,HierMeshEncoder

def prepare_modules(cfg:Config):
    predictor,encoder,decoder = None,None,None
    if cfg.predictor == 'share':
        predictor = SharedSkinWeightPredictors(input_dim=6,num_part=cfg.num_parts,aggr='max',num_layers=cfg.num_layers,use_pool=cfg.pred_use_pool)
    elif cfg.predictor == 'fusing':
        predictor = FusingSkinWeightPredictors(input_dim=6,num_parts=cfg.layer_num_parts,aggr='max',num_layers=cfg.num_layers,use_pool=cfg.pred_use_pool)
    else:
        raise NotImplementedError
    if cfg.encoder == 'share':
        encoder = SharedPartEncoders(input_dim=6,output_dim=cfg.emb_sz,aggr='max',num_layers=cfg.num_layers,use_pool=cfg.enc_use_pool)
    elif cfg.encoder == 'fusing':
        encoder = FusingPartEncoders(input_dim=6,output_dims=cfg.emb_sz,aggr='max',num_layers=cfg.num_layers,use_pool=cfg.enc_use_pool)
    else:
        raise NotImplementedError
    if cfg.decoder == 'share':
        decoder = SharedPartDecoders(input_dim=(np.array(cfg.emb_sz)*2 + 7).tolist(),num_layer=cfg.num_layers)
    elif cfg.decoder == 'fusing':
        decoder = FusingPartDecoders(input_dims=(np.array(cfg.emb_sz)*2 + 7).tolist(),num_layer=cfg.num_layers)
    else:
         raise NotImplementedError
    assert predictor is not None and encoder is not None and decoder is not None
    if cfg.mesh_autoencoder is not None:
        if cfg.mesh_autoencoder == 'naive':
            mesh_encoder = HierMeshEncoder(num_layers=cfg.num_layers,hidden_dims=cfg.mesh_hidden_dims,pool_type=cfg.mesh_pool_type,verbose=True)
            mesh_decoder = HierMeshDecoder(mesh_encoder)
        else:
            raise NotImplementedError
        return predictor,encoder,decoder,mesh_encoder,mesh_decoder
    return predictor,encoder,decoder