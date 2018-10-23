#strt<-Sys.time()
library(clue)
library(plyr)
library(Rcpp)

##### Posterior analysis
burnin = TT/2
################ Relabel: misclassification #######
helpa = log(Pie[,TT])
helpb = t(Beta[,,TT])
helpc = sqrt(sigma2[,TT])
alpbem = .loopexp3(helpa, helpb, helpc, X, Y, J, n)
MCMC_miscla <- list(W = Pie[,(burnin+1):TT], Beta = Beta[,,(burnin+1):TT],
                    Sigma = sigma2[,(burnin+1):TT], ZZ = Z[,(burnin+1):TT])

Zind <- aPiely(alpbem, 1, which.max) #Reference
Zindbem <- Zind
Z0 = array(0, dim=c(n,J))
Zmcmc = array(0, dim=c(n,J))
for (i in 1:J){
  Z0[,i]=(Zind==i)*1
}

for(tt in (burnin+1):(TT-1)){
  helpa = log(Pie[,tt-1])
  helpb = t(Beta[,,tt-1])
  helpc = sqrt(sigma2[, tt-1])
  alpmcmc = .loopexp3(helpa, helpb, helpc, X, Y, J, n)
  Zind <- aPiely(alpmcmc, 1, which.max)
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

# ############## TP ##################
# nTest = length(Ytest)
# nPredic = 8000
# nS <- ncol(MCMC_miscla$W)
# Sprob <- rep(1/nS, nS)
# TP1 <- rep(0, nTest); TP2 <- TP1; TP3 <- TP1; TP4 <- TP1
# Yhat <- rep(0, nTest)
# PI1_0 <- array(0, dim = c(nTest, 2))
# PI2_0 <- array(0, dim = c(nTest, 2))
# PI3_0 <- array(0, dim = c(nTest, 2))
# PI4_0 <- array(0, dim = c(nTest, 2))
# ####### Prediction Interval #####
# for(ii in 1:nTest){  ## testing data
# ### generate predictions
# Ypred <- rep(0, nPredic)
# compS <- sample(1:nS,prob= Sprob,size=nPredic,replace=TRUE)
# for(ss in 1:nPredic){
#   Zprd <- sample(1:J, prob = MCMC_miscla$W[,compS[ss]], size = 1, replace = FALSE)
#   Ypred[ss] <- rnorm(1, Xtest[ii,]%*%MCMC_miscla$Beta[,Zprd,compS[ss]], sqrt(MCMC_miscla$Sigma[Zprd,compS[ss]]))
# }
# #density <- .loopexp4(Y_star, MCMC_miscla$W, Xnew, MCMC_miscla$Beta,  MCMC_miscla$Sigma, ii)
# Yhat[ii] <- mean(Ypred)
# PI1_0[ii,] <- quantile(Ypred, c(.125, .875)) # 75% Predictive interval
# PI2_0[ii,] <- quantile(Ypred, c(.10, .90)) # 80% Predictive interval
# PI3_0[ii,] <- quantile(Ypred, c(.075, .925)) # 85% Predictive interval
# PI4_0[ii,] <- quantile(Ypred, c(.05, .95)) # 90% Predictive interval
# if(Ytest[ii] >= PI1_0[ii,1] && Ytest[ii] <= PI1_0[ii,2]){
#   TP1[ii] <- 1
# }
# if(Ytest[ii] >= PI2_0[ii,1] && Ytest[ii] <= PI2_0[ii,2]){
#   TP2[ii] <- 1
# }
# if(Ytest[ii] >= PI3_0[ii,1] && Ytest[ii] <= PI3_0[ii,2]){
#   TP3[ii] <- 1
# }
# if(Ytest[ii] >= PI4_0[ii,1] && Ytest[ii] <= PI4_0[ii,2]){
#   TP4[ii] <- 1
# }
# #   LP <- quantile(density, 0.025)
# #   UP <- quantile(density, 0.95)
# #   YL <- Y_star[which(density == LP)]
# #   YU <- Y_star[which(density == UP)]
# }

# ############### TN #################
# nTest = length(Ytest1)
# Yhat1 <- rep(0, nTest)
# nS <- ncol(MCMC_miscla$W)
# Sprob <- rep(1/nS, nS)
# TN1 <- rep(0, nTest); TN2 <- TN1; TN3 <- TN1; TN4 <- TN1
# PI1_1 <- array(0, dim = c(nTest, 2))
# PI2_1 <- array(0, dim = c(nTest, 2))
# PI3_1 <- array(0, dim = c(nTest, 2))
# PI4_1 <- array(0, dim = c(nTest, 2))

# ####### Prediction Interval #####
# for(ii in 1:nTest){  ## testing data
# ### generate predictions
# Ypred <- rep(0, nPredic)
# compS <- sample(1:nS,prob= Sprob,size=nPredic,replace=TRUE)
# for(ss in 1:nPredic){
#   Zprd <- sample(1:J, prob = MCMC_miscla$W[,compS[ss]], size = 1, replace = FALSE)
#   Ypred[ss] <- rnorm(1, Xtest1[ii,]%*%MCMC_miscla$Beta[,Zprd,compS[ss]], sqrt(MCMC_miscla$Sigma[Zprd,compS[ss]]))
# }
# #density <- .loopexp4(Y_star, MCMC_miscla$W, Xnew, MCMC_miscla$Beta,  MCMC_miscla$Sigma, ii)
# Yhat1[ii] <- mean(Ypred)
# PI1_1[ii,] <- quantile(Ypred, c(.125, .875)) # 75% Predictive interval
# PI2_1[ii,] <- quantile(Ypred, c(.10, .90)) # 80% Predictive interval
# PI3_1[ii,] <- quantile(Ypred, c(.075, .925)) # 85% Predictive interval
# PI4_1[ii,] <- quantile(Ypred, c(.05, .95)) # 90% Predictive interval
# if(Ytest1[ii] >= PI1_1[ii,1] && Ytest1[ii] <= PI1_1[ii,2]){
#   TN1[ii] <- 1
# }
# if(Ytest1[ii] >= PI2_1[ii,1]&& Ytest1[ii] <= PI2_1[ii,2]){
#   TN2[ii] <- 1
# }
# if(Ytest1[ii] >= PI3_1[ii,1] && Ytest1[ii] <= PI3_1[ii,2]){
#   TN3[ii] <- 1
# }
# if(Ytest1[ii] >= PI4_1[ii,1] && Ytest1[ii] <= PI4_1[ii,2]){
#   TN4[ii] <- 1
# }
# #   LP <- quantile(density, 0.025)
# #   UP <- quantile(density, 0.95)
# #   YL <- Y_star[which(density == LP)]
# #   YU <- Y_star[which(density == UP)]
# }

# #print(Sys.time()-strt)
# PI1 <- array(0, dim = c(length(mark0) + length(mark1), 3))
# colnames(PI1) <- c("lower", "uPieer", "mark")
# PI1[mark1, 3] <- 1
# PI1[mark0, 1:2] <- PI1_0; PI1[mark1, 1:2] <- PI1_1

# PI2 <- PI1; PI3 <- PI1; PI4 <- PI1
# PI2[mark0, 1:2] <- PI2_0; PI2[mark1, 1:2] <- PI2_1
# PI3[mark0, 1:2] <- PI3_0; PI3[mark1, 1:2] <- PI3_1
# PI4[mark0, 1:2] <- PI4_0; PI4[mark1, 1:2] <- PI4_1

