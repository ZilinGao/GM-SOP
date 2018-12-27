# Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks

This is an implementation of GM-SOP([paper](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks.pdf) , 
[supplemental](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks-supplemental.zip))
, created by [Zilin Gao](https://github.com/zilingao) and [Qilong Wang](https://csqlwang.github.io/homepage/).

## Introduction

We propose a global gated mixture second-order pooling network(GM-SOP). 
Our GM-SOP embeds multiple second-order pooling module at the end of CNN which can be trained in an end-to-end manner. 
Compared with [single second-order pooling network](https://github.com/jiangtaoxie/MPN-COV) , GM-SOP break the unimodal distribution assumption.
Besides, in order to solve the high time-consuming brought by multiple second-order pooling modules, a sparsity-constrained gating module is introduced.
GM-SOP is evaluated on two large scale datasets and it is superior to its counterparts, achieving very competitive performance.

## Citation

	@InProceedings{Wang_2018_NeurIPS,
		author = {Wang, Qilong and Gao, Zilin and Xie, Jiangtao and Zuo, Wangmeng and Li, Peihua},
		title = {Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks},
		journal = {Advances in Neural Information Processing Systems (NeurIPS)},
		year = {2018}
	}
	
## DataSets

We evaluated our method on two large-scale datasets: 

[Downsampled ImageNet] (https://arxiv.org/pdf/1707.08819.pdf) (input size 64x64)

Downsampled [Places-365](http://places2.csail.mit.edu/PAMI_places.pdf) 
(we downsample all images to 100x100 by imresize function in matlab with bicubic interpolation method)

## Environment

Our code is implemented with [matconvnet](http://www.vlfeat.org/matconvnet/) toolkit, tested on 1.0-beta25 with Ubuntu 16.04, cuda 10.0.
Considering the dataset is loaded into RAM when the code runs, the workstation should provide available free space as much as the dataset occupied. (Downsampled ImageNet is 13G, Downsampled Places-365 is 45G)

## Results

### Downsampled ImageNet error (%)
| Network             | GFLOPS | Top-1 Error |  Download   |
| ------------------- | ------ | ----------- | ------------|
| ResNet-50 (1x64d)   |  ~4.1  |  23.9        | [Original ResNet-50](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)       |
| ResNeXt-50 (32x4d)  |  ~4.1  |  22.2        | [Download (191MB)](https://s3.amazonaws.com/resnext/imagenet_models/resnext_50_32x4d.t7)       |
| ResNet-101 (1x64d)  |  ~7.8  |  22.0        | [Original ResNet-101](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)      |
| ResNeXt-101 (32x4d) |  ~7.8  |  21.2        | [Download (338MB)](https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_32x4d.t7)      |
| ResNeXt-101 (64x4d) |  ~15.6 |  20.4        | [Download (638MB)](https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_64x4d.t7)       |

## Acknowledgments

* 
* 
* 

