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
  
## DMM for MLP
The folder `dmm` contains three folders: `data`, `model`, and `code`. 
- `data` contains the training set `traing_pdf.npz` and the testing set `testing_pdf.npz` used to train the MLP model. 
- `model` contains the target MLP model that we want to explain. To load the data, model and check the model performance (testing accuracy 99.18%):
```
 python model.py
```

- `code` contains the implementation of the 


## DMM-MEN for CNN


## LEMNA for RNN

## Misc
I maintain a paper list about explanable deep learning. It includes the related paper published in major machine learning conferences (NeurIPS, ICML, and ICLR), security conferences (CCS, USENIX Security, S\&P, and NDSS), and vision conferences (CVPR, ICCV). Please check the following link for more details:
- https://docs.google.com/document/d/11QMlGF1G42v3sRV76ANaFWqdA30jQdEByBh0d8mlx30/edit?usp=sharing
