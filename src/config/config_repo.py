from src.utils import *

hier_base_config = Config(name='hier_base')
# data loading.
hier_base_config.param = ('batch_sz',4)
hier_base_config.param = ('lr',1e-4)
hier_base_config.param = ('epochs',501)
hier_base_config.param = ('part_aug_scale',((0.5, 3), (0.7, 1), (0.5, 1.5)))
hier_base_config.param = ('data_sample_prob1',(0.2, 0.6, 0.2)) # in order: amass, mixamo, rignet.
hier_base_config.param = ('data_sample_prob2',(0.3, 0.4, 0.3)) # in order: amass, mixamo, rignet.
hier_base_config.param = ('num_workers',4)
hier_base_config.param = ('model_path',None)
hier_base_config.param = ('data_num_layers',4)
hier_base_config.param = ('only_layer_id',None)
hier_base_config.param = ('pin_mem',False)
hier_base_config.param = ('prefetch',False)
hier_base_config.param = ('dup_loop_time',1)

# module def.
hier_base_config.param = ('num_layers',4)
hier_base_config.param = ('num_parts',40)
hier_base_config.param = ('emb_sz',128)
hier_base_config.param = ('predictor','share')
hier_base_config.param = ('encoder','share')
hier_base_config.param = ('decoder','share')
hier_base_config.param = ('mesh_autoencoder',None) # None or naive.
hier_base_config.param = ('mesh_hidden_dims',128)
hier_base_config.param = ('mesh_pool_type','naive')
hier_base_config.param = ('pred_use_pool',None)
hier_base_config.param = ('enc_use_pool',None)
hier_base_config.param = ('is_toy',False)

# training strategy.
hier_base_config.param = ('train_mode', 'one_by_one') # one_by_one, prob_sim, step_prob_sim (coming soon).

# valid strategy.
hier_base_config.param = ('valid_interval',20)
hier_base_config.param = ('valid_complete_interval',40) # determine if performing a complete validation for motion sequence instead of several critical poses.
# >>>>> this varies with each model and depends on how to leverage learned pose of the upper mesh.
hier_base_config.param = ('valid_model_mode','org_abl') # only one layer effects, with no strategy provided for lower mesh.
hier_base_config.param = ('valid_mixamo_char_idx',1)
hier_base_config.param = ('valid_mixamo_motions',[10,12])
hier_base_config.param = ('valid_rignet_chars',[380,209,12])
hier_base_config.param = ('valid_amass_poses',[18,273,302])

###### fusing.
fusing_config = Config(name='fusing')
fusing_config.join(hier_base_config)
fusing_config.param = ('num_layers',2)
fusing_config.param = ('train_mode','fusing')
fusing_config.param = ('predictor','fusing')
fusing_config.param = ('encoder','fusing')
fusing_config.param = ('decoder','fusing')
fusing_config.param = ('layer_num_parts',40)
fusing_config.param = ('layer_loss_weights',1)
fusing_config.param = ('layer_var_weight',True)
fusing_config.param = ('layer_final_weights',1) # affect layer-wise interaction when retargeting a higher mesh, when layer var weight is True
fusing_config.param = ('layer_skin_sims',0)
fusing_config.param = ('use_hier_pool',False)
fusing_config.param = ('use_teacher',False)
fusing_config.param = ('external_teacher_path',log_path('org_abl/model_ckpt/ckpt_l0.pth'))
fusing_config.param = ('external_teacher_cfg',None)
fusing_config.param = ('layer_fusing_rates',0.5)
fusing_config.param = ('layer_var_fusing_rate',True)
fusing_config.param = ('layer_final_fusing_rates',0.5)

fusing_config.param = ('valid_model_mode','fusing_spec')
fusing_config.param = ('valid_mixamo_char_idx',1)
fusing_config.param = ('valid_mixamo_motions',[10])
fusing_config.param = ('valid_rignet_chars',[0,3,5,8,11])
fusing_config.param = ('valid_amass_poses',[18,273,302,684,605])
fusing_config.param = ('pooling_ratio',0.6)
fusing_config.param = ('valid_complete_interval',-1)
fusing_config.param = ('valid_interval',100)

fusing_toy_config = Config(name='fusing_toy')
fusing_toy_config.join(fusing_config)
fusing_toy_config.param = ('layer_num_parts',40)
fusing_toy_config.param = ('emb_sz',[128,128])
fusing_toy_config.param = ('batch_sz',2)
fusing_toy_config.param = ('valid_complete_interval',-1)
fusing_toy_config.param = ('valid_interval',1)
fusing_toy_config.param = ('is_toy',True)

# fusing rate?
fusing_frate_64_config = Config(name='fusing_frate_64')
fusing_frate_64_config.join(fusing_config)
fusing_frate_64_config.param = ('layer_fusing_rates',[0.6,0.4])
fusing_frate_64_config.param = ('layer_final_fusing_rates',[0.6,0.4])

# var fusing rate?
fusing_frate_64_var_config = Config(name='fusing_frate_64_var')
fusing_frate_64_var_config.join(fusing_frate_64_config)
fusing_frate_64_var_config.param = ('layer_fusing_rates',[0.4,0.6])