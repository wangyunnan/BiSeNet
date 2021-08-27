# BiSeNet
A pytorch implementation of paper *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation*

## Requirements
- Hardware: PC or Server with two NVIDIA 1080Ti GPUs.
- Software: Ubuntu 16.04, CUDA 10.0, Anaconda3, pytorch 1.1.0

## Dataset
Download Cityscapes dataset [here](https://www.baidu.com/link?url=84aUng-KvWlTVjannp3-F7oYkeVBWPCn0A0iOTVLGZNf0-U5PfG_ggmR5taOJwlW&wd=&eqid=97b2e88b0000c1a60000000561279c76) or wherever convenient for you. Then run script `/datasets/cityscapes/tools/convert_labels.py` to generate trainId from labelId.

## Train
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## Evaluate
```
python evaluate.py
```

## Result
The final mIoU will be around **78.5**, depending on random initialization. In order to confirm the experimental results, [ckeckpoint](https://drive.google.com/file/d/1xlLH8U9AF5D-RcpuXzWdlkKuG84haUpq/view?usp=sharing) (mIoU=78.73) is provided for testing. 
The examples of final result:

<div align=center><img width="512" height="256" src="https://github.com/wangyunnan/BiSeNet/blob/main/save/example/visual/image.png"/></div>

<div align=center><img width="512" height="256" src="https://github.com/wangyunnan/BiSeNet/blob/main/save/example/visual/prediction.png"/></div>
