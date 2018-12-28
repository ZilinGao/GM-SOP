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
		journal = {Neural Information Processing Systems (NeurIPS)},
		year = {2018}
	}
	
## DataSets

We evaluated our method on two large-scale datasets: 

  |Dataset                                                                   |Image Size|Training Set|Validation Set| Class |Download |
  |:------------------------------------------------------------------------:|:--------:|:----------:|:------------:|:-----:|:-------:|
  |[Downsampled ImageNet-1K](https://arxiv.org/pdf/1707.08819.pdf)           |   64x64  |    1.28M   |      50K     |  1000 | Google Drive \| BaiduYun      |
  | Downsampled [Places-365](http://places2.csail.mit.edu/PAMI_places.pdf) **|  100x100 |    1.8M    |     182K     |   365 |\-\-\-       | 
  
  *Work provide the dataset with several partial extracted files, we convert it into mat format for convenient loading in matlab.
  **We downsample all images to 100x100 by _imresize_ function in matlab with _bicubic_ interpolation method.
  
## Environment & Machine Configuration

toolkit: [matconvnet](http://www.vlfeat.org/matconvnet/) 1.0-beta25

matlab: R2016b

system: Ubuntu 16.04

cuda: 9.2

GPU: single GTX 1080Ti

Tips: Considering the whole dataset is loaded into RAM when the code runs, the workstation MUST provide available free space as much as the dataset occupied at least. (Downsampled ImageNet is 13G, Downsampled Places-365 is 45G)
For the same reason, if you want to run with multiple GPUs, RAM should provide dataset_space x GPU_num free space. 
If the RAM is not allowed, you can also restore the data as images in disk and read them from disk during each mini-batch(like most image reading process).

## Start up

The code MUST be compiled by executing vl_compilenn in matlab folder, please see [here](http://www.vlfeat.org/matconvnet/install/) for details. The main function is cnn_imagenet64. 
Considering the long data reading process(about above 1min), we provide a tiny fake data mat file: examples/GM/imdb.mat as default setting for quick debug. 
If you want to train model, please modify the dataset file path by changing opts.imdbPath in function cnn_imagenet64.


## Results

### Downsampled ImageNet-1K

|          Network         | Parameters |  Dimension | Top-1 error / Top-5 error (%)|
|:-------------------------|:----------:|:----------:|:----------------------------:|
| ResNet-18                |    0.9M    |    128     |        52.00/26.97           |
| ResNet-18-512d           |    1.3M    |    512     |        49.08/24.25           |
| ResNet-50                |    2.4M    |    128     |        43.28/19.39           |
| ResNet-50-8256d          |   11.6M    |    8256    |        41.42/18.14           |
| GM-GAP-16-8 + ResNet-18  |   2.3M     |    512     |        42.37/18.82           |
| GM-GAP-16-8 + ResNet-18* |   2.3M     |    512     |        40.03/17.91           |
| GM-GAP-16-8 + WRN-36-2   |   8.7M     |    512     |        35.97/14.41           | 
| GM-SOP-16-8 + ResNet-18  |  10.3M     |    8256    |        38.21/17.01           | 
| GM-SOP-16-8 + ResNet-50  |  11.9M     |    8256    |        35.73/14.96           | 
| GM-SOP-16-8 + WRN-36-2   |  15.7M     |    8256    |        32.33/12.35           | 

*denotes double number of training images including original images AND their horizontal flip ones


### Downsampled Places-365 

|       Network     | Dimension | Top-1 error (%) | Top-5 error (%)|
|:------------------|:---------:|:---------------:|:--------------:|
| ResNet-18-512d    |    512    |      49.96      |    19.19       |
| GM-GAP-16-8       |    512    |      48.07      |    17.84       |
| ResNet-18-8256d   |   8256    |      49.99      |    19.32       |
| SR-SOP            |   8256    |      48.11      |    18.01       |
| Parametric SR-SOP |   8256    |      47.48      |    17.52       |
| GM-SOP-16-8       |   8256    |      47.18      |    17.02       | 


## Acknowledgments

* The authors thank pioneering work: [MPN-COV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Is_Second-Order_Information_ICCV_2017_paper.pdf),
[iSQRT-COV](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf) and thank them for providing packaged high-efficiency code.
* We would like to thank MatConvNet team for developing MatConvNet toolbox.

