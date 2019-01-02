# Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks

This is an implementation of GM-SOP([paper](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks.pdf) , 
[supplemental](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks-supplemental.zip))
, created by [Zilin Gao](https://github.com/zilingao) and [Qilong Wang](https://csqlwang.github.io/homepage/).

## Introduction

In most of existing deep convolutional neural networks (CNNs) for classification,
global average (first-order) pooling (GAP) has become a standard module to summarize
activations of the last convolution layer as final representation for prediction.
Recent researches show integration of higher-order pooling (HOP) methods clearly
improves performance of deep CNNs. However, both GAP and existing HOP
methods assume unimodal distributions, which cannot fully capture statistics of
convolutional activations, limiting representation ability of deep CNNs, especially
for samples with complex contents. To overcome the above limitation, this paper
proposes a global Gated Mixture of Second-order Pooling (GM-SOP) method to
further improve representation ability of deep CNNs. To this end, we introduce
a sparsity-constrained gating mechanism and propose a novel parametric SOP as
component of mixture model. Given a bank of SOP candidates, our method can
adaptively choose Top-K(K > 1) candidates for each input sample through the
sparsity-constrained gating module, and performs weighted sum of outputs of K
selected candidates as representation of the sample. The proposed GM-SOP can
flexibly accommodate a large number of personalized SOP candidates in an efficient
way, leading to richer representations. The deep networks with our GM-SOP can be
end-to-end trained, having potential to characterize complex, multi-modal distributions.
The proposed method is evaluated on two large scale image benchmarks (i.e.,
downsampled ImageNet-1K and Places365), and experimental results show our
GM-SOP is superior to its counterparts and achieves very competitive performance.


## Citation

	@InProceedings{Wang_2018_NeurIPS,
		author = {Wang, Qilong and Gao, Zilin and Xie, Jiangtao and Zuo, Wangmeng and Li, Peihua},
		title = {Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks},
		journal = {Neural Information Processing Systems (NeurIPS)},
		year = {2018}
	}
	
## Datasets

We evaluated our method on two large-scale datasets: 

  |Dataset                   |Image Size|Training Set|Validation Set| Class |Download |
  |:------------------------:|:--------:|:----------:|:------------:|:-----:|:-------:|
  |Downsampled ImageNet-1K*  |   64x64  |    1.28M   |      50K     |  1000 | Google Drive* \| [BaiduYun(13G)](https://pan.baidu.com/s/1FwupydRfZ4hY7UnHeuv3Qw)\*      |
  | Downsampled Places-365 **|  100x100 |    1.8M    |     182K     |   365 |----     | 
  
  *The work[[arxiv]](https://arxiv.org/pdf/1707.08819.pdf) provides a downsampled version of
ImageNet-1K dataset. In this work, each image in ImageNet dataset (including both training set and validation set) is downsampled by _box sampling_ method to the size of 64x64, resulting in a downsampled ImageNet-1K dataset with same quantity samples and lower resolution. As it descripted, downsampled ImageNet-1K dataset might represent a viable alternative to the CIFAR datasets while dealing with more complex data and classes. 
<br>Based on above work, we prepare one copy of downsampled ImageNet-1K in _.mat_ form for public use. To be specific, on each part of original downsampled ImageNet-1K dataset file, we use _unpickle_ function in python enviroment followed with _scipy.io.savemat_ to convert the original file into _.mat_ format, finally concatenate all parts into one full _.mat_ file.
<br>MD5code: fe50ac93f74744b970b3102e14e69768
  
  **We downsample all images to 100x100 by _imresize_ function in matlab with _bicubic_ interpolation method.

## Environment & Machine Configuration

toolkit: [matconvnet](http://www.vlfeat.org/matconvnet/) 1.0-beta25

matlab: R2016b

cuda: 9.2

GPU: single GTX 1080Ti

system: Ubuntu 16.04

RAM: 32G

Tips: Considering the whole dataset is loaded into RAM when the code runs, the workstation MUST provide available free space as much as the dataset occupied at least. Downsampled ImageNet is above 13G, we use the machine equipped with 32G RAM for experiments.
For the same reason, if you want to run with multiple GPUs, RAM should provide _dataset_space x GPU_num_ free space. 
If the RAM is not allowed, you can also restore the data in form of image files in disk and read them from disk during each mini-batch(like most image reading process).

## Start up

The code MUST be compiled by executing _matlab/vl_compilenn.m_, please see [here](http://www.vlfeat.org/matconvnet/install/) for details. The main function is _example/GM/cnn_imagenet64.m_ . 
Considering the long data reading process(about above 1min), we provide a tiny FAKE data mat file: _examples/GM/imdb.mat_ as default setting for quick debug. 
If you want to train model, please download the full dataset we provide above and modify the dataset file path by changing _opts.imdbPath_ in function _example/GM/cnn_imagenet64.m_.

## Results

### Downsampled ImageNet-1K

|          Network         | Parameters |  Dimension | Top-1 error / Top-5 error (%)| Model |
|:-------------------------|:----------:|:----------:|:----------------------------:|:-----:|
| ResNet-18                |    0.9M    |    128     |        52.00/26.97           | ------|
| ResNet-18-SR-SOP         |    9.0M    |    8256    |        40.32/18.94           | ------|
| GM-GAP-16-8 + ResNet-18  |   2.3M     |    512     |        42.25/19.46           | Google Drive \| [BaiduYun](https://pan.baidu.com/s/1fmmT0haaqvG2uGRqoyp8Yw)|
| GM-GAP-16-8 + WRN-36-2   |   8.7M     |    512     |        35.97/14.41           | Coming soon!|
| GM-SOP-16-8 + ResNet-18  |  10.3M     |    8256    |        38.21/17.01           | Coming soon!|
| GM-SOP-16-8 + WRN-36-2   |  15.7M     |    8256    |        32.33/12.35           | Coming soon!|

- MD5 code: 
<br>GM-GAP-16-8 + ResNet-18: f80738566ffe9cabb7a1e88ea6c79dcf

## Acknowledgments

* The authors thank pioneering work: [MPN-COV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Is_Second-Order_Information_ICCV_2017_paper.pdf),
[iSQRT-COV](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf) and thank them for providing packaged high-efficiency code.
* We would like to thank MatConvNet team for developing MatConvNet toolbox.

## Contact Information

If you have any suggestion or question, you can leave a message here or contact us directly: gzl@mail.dlut.edu.cn . Thanks for your attention!
