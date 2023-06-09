

![repo_title](demo/repo_title.png)

***

![Python](https://img.shields.io/badge/Python->=3.7-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.5.0-Red?logo=pytorch) ![PyG](https://img.shields.io/badge/PyG->=2.2.0-Red?logo=pyg) [![arXiv](https://img.shields.io/badge/arxiv-2303.10941-green.svg)](https://arxiv.org/abs/2303.10941) [![GitHub license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://github.com/HoeTosaki/HMC_repo/blob/main/LICENSE) ![visitor badge](https://api.visitor.plantree.me/visitor-badge/pv?namespace=hoewang.hmc&key=default&color=orange&style=plastic&label=visitors%20of%20HMC)

**Official Repository** for the newly published paper entitled "**HMC: Hierarchical Mesh Coarsening for Skeleton-free Motion Retargeting**"

![title_img](demo/title_img.jpg)

[[Project Page]](https://semanticdh.github.io/HMC/)  [[ArXiv Page]](https://arxiv.org/abs/2303.10941)  [[Paper]](https://arxiv.org/pdf/2303.10941.pdf) 

***

## Demo Presentation

<!-- <video  controls="controls" loop="loop">
    <source src="demo/videos.mp4" type="video/mp4">
</video> -->

## Environment Preparation

### Conda

```shell
conda install --yes --file requirements.txt
```

### Pip

```shell
pip install -r requirements.txt
```

## Quick Start

#### 1. download the [pretrained model](https://drive.google.com/file/d/1RPkzrJQLBIMGHyeqkbMVQcU61QlEAZuK/view?usp=sharing) & [demo data](https://drive.google.com/file/d/18G4k1n3lVhnff9d_zI3PbCv7h-_cQKUs/view?usp=sharing) from Google Drive

Drop the pretrained model into `pretrained/` and demo data into `data/`, respectively.

Here are backup links to Baidu Disk for both [model](https://pan.baidu.com/s/1zqTHhhrPJbksPZRnBgt4pA?pwd=2023) & [data](https://pan.baidu.com/s/1CbSELsD3FcYmk8x-eAenaQ?pwd=2023).

#### 2.  retarget greeting motion to a Mixamo character in T-pose.
Run
```shell
python inference_hmc.py
```

Then, a motion sequence as `greeting_on_target-XXXXXX.obj` will be saved in `data\greeting_on_target\`.

If the retargeted sequence is converted to `.abc` format (a routine for automatic conversion will be provided in the future), it should be like this:

<img src="demo/demo_output.gif" alt="demo_output" style="zoom: 33%;" />

## Inference on your own data

#### 1. prepare a source motion

Create a source folder `data/{src_name}` comprising source T-pose (`data/{src_name}/{src_name}-tpose.obj`) and motion sequence (`data/{src_name}/{src_name}-{idx}.obj`, where `{idx}` is counted from $1$). 

#### 2. prepare a target character

Create a target folder `data/{tgt_name}` comprising only target T-pose (`data/{tgt_name}/{tgt_name}-tpose.obj`).

#### 3. retarget motions

Run

```shell
python inference_hmc.py --src_name={src_name} --tgt_name={tgt_name}
```

and a motion sequence of the same length as source motion will be produced in folder `data/{tgt_name}/`

#### X. to accelerate inference or improve retargeting performance

As an alternative to 3, run

```python
python inference_hmc.py --src_name={src_name} --tgt_name={tgt_name} --precoarsen_src={pc_ratio}
```

where `{pc_ratio}` is a continuous value in $(0,1]$ that pre-coarsens the input source motion before retargeting. With a smaller `{pc_ratio}`, the retargeting process can be accelerated, and the model also considers few mesh details on the source. However, one should note that in some cases, a small `{pc_ratio}` may induce extra jitters in target motion due to uncaught local motions.


 ## Citation

If you use HMC in any context, please cite the following paper:

```
@misc{wang2023hmc,
      title={HMC: Hierarchical Mesh Coarsening for Skeleton-free Motion Retargeting}, 
      author={Haoyu Wang and Shaoli Huang and Fang Zhao and Chun Yuan and Ying Shan},
      year={2023},
      eprint={2303.10941},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
  }
```

***

<a href="https://info.flagcounter.com/mRXd"><img src="https://s11.flagcounter.com/count/mRXd/bg_3F90EB/txt_FFFFFF/border_CCCCCC/columns_8/maxflags_12/viewers_Visitors+of+HMC+repo/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
