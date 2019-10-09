library(clue)
library(plyr)
library(Rcpp)
library(R.matlab)
library(coda)
library(Rcpp)
set.seed(1567)

################ Generate test data #######
Xtest <- X[1:1000, ]
Ytest <- Y[1:1000]

print (nrow(Xtest))
print (length(Ytest))
print (ncol(Xtest))

##### Posterior analysis
burnin = TT/2
print ("################ Relabel: misclassification #######")
helpa = log(Pie[,TT])
helpb = t(Beta[,,TT])
helpc = sqrt(sigma2[,TT])
Rcpp::sourceCpp('express.cc')
alpbem = .loopexp3(helpa, helpb, helpc, X, Y, J, n) # compute the probability of each sample in each class
MCMC_miscla <- list(W = Pie[,(burnin+1):TT], Beta = Beta[,,(burnin+1):TT],
                    Sigma = sigma2[,(burnin+1):TT], ZZ = Z[,(burnin+1):TT])

Zind <- apply(alpbem, 1, which.max) #Reference
Zindbem <- Zind
Z0 = array(0, dim=c(n,J))
Zmcmc = array(0, dim=c(n,J))
for (i in 1:J){
  Z0[,i]=(Zind==i)*1
}

for(tt in (burnin+1):(TT-1)){
  print (tt)
  helpa = log(Pie[,tt-1])
  helpb = t(Beta[,,tt-1])
  helpc = sqrt(sigma2[, tt-1])
  alpmcmc = .loopexp3(helpa, helpb, helpc, X, Y, J, n)
  Zind <- apply(alpmcmc, 1, which.max)
  for (i in 1:J){
    Zmcmc[,i]=(Zind==i)*1
  }
  dist = t(Z0)%*%abs(1-Zmcmc);
  reorder = as.vector(solve_LSAP((dist), maximum = FALSE))
  MCMC_miscla$W[,(tt-burnin)] <-  Pie[reorder,tt]
  MCMC_miscla$Beta[,,(tt-burnin)] <-  Beta[,reorder,tt]
  MCMC_miscla$Sigma[,(tt-burnin)] <- sigma2[reorder,tt]
  MCMC_miscla$ZZ[,(tt-burnin)] <- mapvalues(Z[,tt], from = 1:J, to = reorder)
}

print ("############## predict ##################")
nTest = length(Ytest)
nPredic = 8000
nS <- ncol(MCMC_miscla$W) # the number of iteration contribute to this results
Sprob <- rep(1/nS, nS) # set each iteration equal probability
TP1 <- rep(0, nTest); TP2 <- TP1; TP3 <- TP1; TP4 <- TP1
Yhat <- rep(0, nTest)
PI1_0 <- array(0, dim = c(nTest, 2))
PI2_0 <- array(0, dim = c(nTest, 2))
PI3_0 <- array(0, dim = c(nTest, 2))
PI4_0 <- array(0, dim = c(nTest, 2))
####### Prediction Interval #####
for(ii in 1:nTest){  ## testing data
  ### generate predictions
  Ypred <- rep(0, nPredic)
  compS <- sample(1:nS,prob= Sprob,size=nPredic,replace=TRUE)
  for(ss in 1:nPredic){
    Zprd <- sample(1:J, prob = MCMC_miscla$W[,compS[ss]], size = 1, replace = FALSE)
    Ypred[ss] <- rnorm(1, Xtest[ii,]%*%MCMC_miscla$Beta[,Zprd,compS[ss]], sqrt(MCMC_miscla$Sigma[Zprd,compS[ss]]))
  }
  #density <- .loopexp4(Y_star, MCMC_miscla$W, Xnew, MCMC_miscla$Beta,  MCMC_miscla$Sigma, ii)
  Yhat[ii] <- mean(Ypred)
  PI1_0[ii,] <- quantile(Ypred, c(.125, .875)) # 75% Predictive interval
  PI2_0[ii,] <- quantile(Ypred, c(.10, .90)) # 80% Predictive interval
  PI3_0[ii,] <- quantile(Ypred, c(.075, .925)) # 85% Predictive interval
  PI4_0[ii,] <- quantile(Ypred, c(.05, .95)) # 90% Predictive interval
  if(Ytest[ii] >= PI1_0[ii,1] && Ytest[ii] <= PI1_0[ii,2]){
    TP1[ii] <- 1
  }
  if(Ytest[ii] >= PI2_0[ii,1] && Ytest[ii] <= PI2_0[ii,2]){
    TP2[ii] <- 1
  }
  if(Ytest[ii] >= PI3_0[ii,1] && Ytest[ii] <= PI3_0[ii,2]){
    TP3[ii] <- 1
  }
  if(Ytest[ii] >= PI4_0[ii,1] && Ytest[ii] <= PI4_0[ii,2]){
    TP4[ii] <- 1
  }
#   LP <- quantile(density, 0.025)
#   UP <- quantile(density, 0.95)
#   YL <- Y_star[which(density == LP)]
#   YU <- Y_star[which(density == UP)]
}

p_original = exp(Ytest)/(exp(Ytest)+1)
p_GMM = exp(Yhat)/(exp(Yhat)+1)
print ("error")
error = norm((p_original - p_GMM), type = '2')
print (error)
print ("RMSD")
RMSD = sqrt(sum((p_original - p_GMM)^2)/1000)
print (RMSD)

################ final parameters ################
Pi_final <- rowMeans(MCMC_miscla$W)
Sigma2_final <- rowMeans(MCMC_miscla$Sigma)
Beta_final <- rowMeans(MCMC_miscla$Beta, dims = 2)
helpa_final <- log(Pi_final)
helpb_final <- t(Beta_final)
helpc_final <- sqrt(Sigma2_final)

Rcpp::sourceCpp('express.cc')
alpmcmc <- .loopexp3(helpa_final, helpb_final, helpc_final, X, Y, J, n)
Zind_final <- apply(alpmcmc, 1, which.max)
n <- length(Zind_final)

################# plot the change of different component of Beta ################
#name <- paste0("feature", ".pdf")
#pdf(name,  width=14, height=5)
#for (i in 1:J){
#  Beta_tmp <- rowMeans(MCMC_miscla$Beta[ ,i, ])
#  plot(Beta_tmp, xlab = "Feature", ylab = paste0("component_",i), type = 'p') 
#}
#dev.off() 

#name = paste0("Clusters", ".pdf")
################# final clustering ################
#pdf(name,  width=7, height=7)
#hist(Zind_final)
#dev.off() 
################ feature selection ################

final_params <- list(Z = Zind_final, Beta = Beta_final, Sigma2 = Sigma2_final)
save(final_params, file = 'dmm_parameters.RData')