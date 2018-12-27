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

[Downsampled ImageNet](https://arxiv.org/pdf/1707.08819.pdf) (input size 64x64)

Downsampled [Places-365](http://places2.csail.mit.edu/PAMI_places.pdf) 
(we downsample all images to 100x100 by imresize function in matlab with bicubic interpolation method)

## Environment

Our code is implemented with [matconvnet](http://www.vlfeat.org/matconvnet/) toolkit, tested on 1.0-beta25 with Ubuntu 16.04, cuda 10.0.
Considering the dataset is loaded into RAM when the code runs, the workstation should provide available free space as much as the dataset occupied. (Downsampled ImageNet is 13G, Downsampled Places-365 is 45G)

## Results

### Downsampled ImageNet error (%)

|          Network         | Parameters |  Dimension | Top-1 error / Top-5 error (%)|
| -------------------------| ---------- | ---------- | -----------------------------|
| ResNet-18-512d           |    1.3M    |    512     |        49.08/24.25           |
| ResNet-18-8256d          |    ----    |    8256    |        47.29/-----           |
| ResNet-50-8256d          |  11.6M     |    8256    |        41.42/18.14           |
| GM-GAP-16-8 + ResNet-18  |   2.3M     |    512     |        42.37/18.82           |
| GM-GAP-16-8 + ResNet-18* |   2.3M     |    512     |        40.03/17.91           |
| GM-GAP-16-8 + WRN-36-2   |   8.7M     |    512     |        35.97/14.41           | 
| GM-SOP-16-8 + ResNet-18  |  10.3M     |    8256    |        38.21/17.01           | 
| GM-SOP-16-8 + ResNet-50  |  11.9M     |    8256    |        35.73/14.96           | 
| GM-SOP-16-8 + WRN-36-2   |  15.7M     |    8256    |        32.33/12.35           | 

*denotes double number of training images including original images and their horizontal flip ones


### Downsampled Places-365 error (%)

|       Network     | Dimension | Top-1 error (%) | Top-5 error (%)|
| ------------------| --------- | --------------- | ---------------|
| ResNet-18-512d    |    512    |      49.96      |    19.19       |
| GM-GAP-16-8       |    512    |      48.07      |    17.84       |
| ResNet-18-8256d   |   8256    |      49.99      |    19.32       |
| SR-SOP            |   8256    |      48.11      |    18.01       |
| Parametric SR-SOP |   8256    |      47.48      |    17.52       |
| GM-SOP-16-8       |   8256    |      47.18      |    17.02       | 


## Acknowledgments

* 
* 
* 

