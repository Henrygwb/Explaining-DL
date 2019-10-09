# Black-box explanation of deep learning models using mixture regression models

## Introduction

This repo is about explaining the outputs of a well-trained black-box deep learning model, including MLP, CNN, and RNN. To be specific, given a testing sample and the output probability of a well-trained model on that sample, our techniques could identify a set of important features that make key contributions to that output. The high-level idea of these techniques is to approximate the complicated deep learning model with a piece-wise linear model - mixture regression model and add different regularization terms based on the input attributions of different types of models (i.e., CNN with an image as input - elastic net [[1]](http://www.personal.psu.edu/wzg13/publications/nips18.pdf), RNN with a sequence as input - fused lasso [[2]](http://www.personal.psu.edu/wzg13/publications/ccs18.pdf)). 

More details can be found in our papers:

```
[1] [Explaining Deep Learning Models -- A Bayesian Non-parametric Approach]
Wenbo Guo, Sui Huang, Yunzhe Tao, Xinyu Xing, Lin Lin
In NeurIPS 2018

[2] LEMNA: Explaining Deep Learning based Security Applications 
Wenbo Guo, Dongliang Mu, Jun Xu, Purui Su, Gang Wang, Xinyu Xing 
In CCS 2018
```

The repo consists the following three parts: 
  - Dirichlet process mixture regression model (DMM). This model is used to explainan MLP. We use an example of a PDFmalware classifier to demonstrate its usage.
  - Dirichlet process mixture regression model with multiple elastic nets (DMM-MEN) with an example of explaining CNN image classifier.
  - Mixture regression model with fused lasso (LEMNA) with an example of explaining and patching a binary function start identification RNN. 

## Requirments

`R` and `python` are required. I am using `R 3.4.3` and `python 2.7`, other versions may also work. Some required R packages are specified in each R script. To install them, use `install.packages(package_name)`. The required python libraries are also specified in the python file, they can be installed via either `pip` or `conda`. Note that I am using `rpy==2.8.6` and `Keras==2.2.4` .


## DMM for MLP

The folder `dmm` contains three folders: `data`, `model`, and `code`. 
- `data` contains the training set `traing_pdf.npz` and the testing set `testing_pdf.npz` used to train the MLP model. 
- `model` contains the target MLP model that we want to explain. 
- `code` contains the implementation of the explanation and evaluation process. 

The whole work-flow is as follows:
- Load the data, model; check the model performance (testing accuracy 99.18%); and get the model predictions for the a group samples (either testing or training samples).
	
```
python model.py
```

- Use the selected samples and the corresponding DL model predictions to fit a DMM model and get the regressions coefficients of the trained DMM model.
		
```
Rscript dmm.r
```
You should be able to get a `dmm_parameters.RData` file which stores the final regression parameters and The terminal should print the final RMSE: XXXX.

- Pinpoint the important features by ranking the regression coefficients and conduct the fidelity tests (feature deduction, feature augmentation, and Synthetic test):
		
```
python xai_mlp_dmm.py -nf 5
```

`-nf` controls the number of features selected. The final printed information is the three fidelity testing results of our technique and random feature selection.

Note that the python file `xai_dmm.py` calls the R functions that fitting the DMM model (`dmm.R`) and conducting the post-processing (`analysis.R`), if you want to tune the hyper-parameters of the DMM model, you can change them in the `dmm.R` (I put the comments to locating the hyper-parameter initialization.). If you encounter errors related to the R code, it is likely you don't install the required packages or you don't make the names of the input samples consistent.

## DMM-MEN for CNN


## LEMNA for RNN

## Misc

I maintain a paper list about explanable deep learning. It includes the related paper published in major machine learning conferences (NeurIPS, ICML, and ICLR), security conferences (CCS, USENIX Security, S\&P, and NDSS), and vision conferences (CVPR, ICCV). Please check the following link for more details:
- https://docs.google.com/document/d/11QMlGF1G42v3sRV76ANaFWqdA30jQdEByBh0d8mlx30/edit?usp=sharing
