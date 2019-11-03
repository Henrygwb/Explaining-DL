# Black-box explanation of deep learning models using mixture regression models

## Introduction

This repo is about explaining the outputs of a well-trained black-box deep learning model, including MLP, CNN, and RNN. To be specific, given a testing sample and the output probability of a well-trained model on that sample, our techniques could identify a set of important features that make key contributions to that output. The high-level idea of these techniques is to approximate the complicated deep learning model with a piece-wise linear model - mixture regression model and add different regularization terms based on the input attributions of different types of models (i.e., CNN with an image as input - elastic net [[1]](http://www.personal.psu.edu/wzg13/publications/nips18.pdf), RNN with a sequence as input - fused lasso [[2]](http://www.personal.psu.edu/wzg13/publications/ccs18.pdf)). 

More details can be found in our papers:

```
[1] Explaining Deep Learning Models -- A Bayesian Non-parametric Approach
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

`R` and `python` are required. I am using `R 3.4.3` and `python 2.7`, other versions may also work. Some required R packages are specified in each R script. To install them, use `install.packages(package_name)`. The required python libraries are also specified in the python file, they can be installed via either `pip` or `conda`. Note that I am using `rpy==2.8.6` and `Keras==2.2.4` . Two R packages are required and will be loaded by python `genlasso` and `gsubfn`.


## DMM for MLP

The folder `dmm` contains four folders: `data`, `model`, `results` and `code`. 
- `data` contains the training set `traing_pdf.npz` and the testing set `testing_pdf.npz` used to train the MLP model. 
- `model` contains the target MLP model that we want to explain. 
- `code` contains the implementation of the explanation and evaluation process. 
- `results` has two groups of dmm model parameters can be used for extracting important features and fidelity tests.

The whole work-flow is as follows:
- Load the data, model; check the model performance (testing accuracy 99.18%); and get the model predictions for the a group samples (either testing or training samples).
	
```
python mlp_malware.py
```

- Use the selected samples and the corresponding DL model predictions to fit a DMM model and get the regressions coefficients of the trained DMM model.
		
```
Rscript dmm.r
```
You should be able to get a `dmm_parameters.RData` file which stores the final regression parameters and The terminal should print the final RMSE.

- Pinpoint the important features by ranking the regression coefficients and conduct the fidelity tests (feature deduction, feature augmentation, and Synthetic test) [2]:
		
```
python xai_mlp_dmm.py -nf 5
```

`-nf` controls the number of features selected. The final printed information is the three fidelity testing results of our technique and random feature selection.

The DMM model is in `dmm.R` and the post-processing is `analysis.R`, if you want to tune the hyper-parameters of the DMM model, you can change them in the `dmm.R` (I put the comments to locating the hyper-parameter initialization.). If you encounter errors related to the R code, it is likely you don't install the required packages or you don't make the names of the input samples consistent.

## DMM-MEN for CNN

The folder `dmm-men` contains the model of DMM-MEN and an example of explaining the Trouser class of a CNN trained on Fashion-MNIST model. It shares the similar structure with `dmm`.

- Load the data, model; get the model predictions for the samples belonging to the target class and conduct dimensional reduction.
	
```
python cnn_fashion.py
```

- Fit a DMM-MEN model.
		
```
Rscript dmm_men.r
```
Model parameters are stored in `results`.

- Pinpoint the important features and visualize them in heatmap; Conduct the bootstrap fidelity tests in [1]:
		
```
python xai_cnn_dmm_men.py
```

Here, I conducted a global approximation for the target class. Since the samples are assigned to multiple mixture components, we get multiple groups of common rules (commonly important features). The fidelity tests are conducted on the component which is assigned with the most number of samples. I put my results in the `results` folder together with some visualization of the generated testing samples.

## LEMNA for RNN

The folder `lemna` contains the model of Fused LASSO regression and an example of explaining an RNN trained to identify the function starts of binaries complied with O1 optimization option [2]. The codes include training and loading the RNN model, explanation and fidelity testing, and model patching. 

- Load the data, model; get the model predictions for the samples belonging to the target class and conduct dimensional reduction.
	
```
python rnn_binary.py
```

- Explaining with LENMA and conduct fidelity tests.
		
```
python lemna_beta.py
```

- Patching the testing errors of the RNN model based on explanation.

```
python patch.py
```

Hyperparameters of each component can be set in each python file and I wrote comments to explain the designed classes and functions. I also put my explanation results and the fixed model in the `results` folder. The fixed model is able to reduce the number of false positive from 33 to 17 and the number of false negative from 48 to 24.

## Misc

We maintain a paper list about explanable deep learning. It includes the related papers published in major machine learning conferences (NeurIPS, ICML, and ICLR), security conferences (CCS, USENIX Security, S\&P, and NDSS), and vision conferences (CVPR, ICCV). Please check the following link for more details:
- https://docs.google.com/document/d/11QMlGF1G42v3sRV76ANaFWqdA30jQdEByBh0d8mlx30/edit?usp=sharing

## Acknowlegdement
We thank Zhenyu Hu from Texas A&M University for his help in building this repository. 
