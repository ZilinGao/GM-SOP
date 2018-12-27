# Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks

This is an implementation of GM-SOP([paper](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks.pdf) , [supplemental](https://papers.nips.cc/paper/7403-global-gated-mixture-of-second-order-pooling-for-improving-deep-convolutional-neural-networks-supplemental.zip)).
It is created by [Zilin Gao](https://github.com/zilingao) and [Qilong Wang](https://csqlwang.github.io/homepage/).
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/fast_MPN-COV.JPG" width="80%"/>
</div>

## Introduction

We propose a global gated mixture second-order pooling network(GM-SOP). 
Our GM-SOP embeds multiple second-order pooling module at the end of CNN which can be trained in an end-to-end manner. 
Compared with [single second-order pooling module](https://github.com/jiangtaoxie/MPN-COV) , GM-SOP break the unimodal distribution assumption.
Besides, in order to solve the high time-consuming brought by multiple second-order pooling modules, a sparsity-constrained gating module is introduced.
GM-SOP is evaluated on two large scale datasets and it is superior to its counterparts, achieving very competitive performance.

### Citation

	@InProceedings{Wang_2018_NeurIPS,
		author = {Wang, Qilong and Gao, Zilin and Xie, Jiangtao and Zuo, Wangmeng and Li, Peihua},
		title = {Global Gated Mixture of Second-order Pooling for Improving Deep Convolutional Neural Networks},
		journal = {Advances in Neural Information Processing Systems (NeurIPS)},
		year = {2018}
	}
	
### DataSets



### Environment

Our code is implemented with [matconvnet](http://www.vlfeat.org/matconvnet/) toolkit, tested on 1.0-beta25 with Ubuntu 16.04, cuda 10.0.
Considering the dataset is loaded into RAM when the code runs, the workstation should provide available free space as much as the dataset occupied. (Downsampled ImageNet is 13G, Downsampled Places-365 is 45G)

### Results

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo




## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

