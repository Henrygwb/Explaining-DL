setwd("C:/Users/zheny/Documents/GitHub/Explaining-DL/dmm/code")
set.seed(1567)

# Install and import the necessary packages.
#install.packages(MASS)
#install.packages(mvtnorm)
#install.packages(MCMCpack)
#install.packages(Rcpp)
#install.packages(R.matlab)

library(MASS)
library(mvtnorm)
library(MCMCpack)
library(Rcpp)
library(R.matlab)
sourceCpp("express.cc")

print ('Loading data... ')
data <- readMat('../data/data_GMM.mat')
X <- data[['X']]
Y <- data[['y']]
n <- nrow(Y)
ss1 <- as.integer(n*0.8)

print ('Training... ')
for(iter in 1:1){
  cat("iter = ", iter, "\n")
  strt<-Sys.time()
  ssel <- sample(n, ss1)
  tmp <- 1:n; tmp_1 <- tmp[ssel]; tmp_2 <- tmp[-ssel]
  Ytest <- Y[tmp_2, ]
  Xtest <- X[tmp_2, ]
  Y <- Y[tmp_1, ]
  X <- X[tmp_1, ]
  n <- nrow(X)
  p <- ncol(X)
  
  ###### Seting hyperparameters for prior
  ### (beta, sigma2) ~ NIG(m, V_beta, a,b)
  V_beta <- diag(3,p)
  m <- colMeans(X)#rep(0,p)#colMeans(X)
  J = 3;
  e = 5;
  f = 1; #alpha
  TT = 20000;
  b = 1/2
  a = 1/2
  temp3 = solve(V_beta)
  
  ###### Initialize parameters for MCMC
  alpha <- rep(0, TT); alpha[1] <- e
  nu <- array(0, dim = c(J-1, TT))
  nu[,1] <- rbeta(J-1,1,alpha[1])
  
  Pie <- array(0, dim = c(J, TT))
  Pie[1,1] <- nu[1,1]
  tmp <- 1-nu[1,1]
  for(jj in 2:(J-1)){
    Pie[jj,1] = tmp * nu[jj,1]
    tmp <- tmp * (1-nu[jj,1])
  }
  Pie[J,1] <- tmp
  
  Beta <- array(0, dim = c(p,J,TT))
  sigma2 <- array(0, dim=c(J,TT))
  for(j in 1:J){
    sigma2[j, 1] <- rinvgamma(1,a,b) #1 /rgamma(1, a, b)
    Beta[, j, 1] <- mvrnorm(1, m, V_beta * sigma2[j, 1])
  }
  Z <- array(0,dim=c(n,TT))
  
  for (tt in 2 : TT){
    ## print status
    if (tt %% 100 == 0) {
      cat("step = ", tt, "\n")
    }
    
    ### update Z
    helpa = log(Pie[,tt-1])
    helpb = t(Beta[,,tt-1])
    helpc = sqrt(sigma2[, tt-1])
    
    Z[,tt] = .loopexp2(helpa, helpb, helpc, X, Y, J, n)
    
    
    ### update nu
    for(j in 1:(J-1)){
      aj <- sum(Z[,tt] == j)
      as <- sum(Z[,tt] >j)
      nu[j,tt] <- rbeta(1, aj + 1, alpha[tt-1] + as)
    }
    
    
    ### update Pie
    Pie[1,tt] <- nu[1,tt]
    tmp <- 1-nu[1,tt]
    for(jj in 2:(J-1)){
      Pie[jj,tt] = tmp * nu[jj,tt]
      tmp <- tmp * (1-nu[jj,tt])
    }
    Pie[J,tt] <- tmp
    
  
    ### update alpha
    alpha[tt] <- rgamma(1,J+e-1, rate = f- sum(log(1-nu[,tt])))
    
    
    ### update mu and Sigma
    for(j in 1:J){
      aj = sum(Z[,tt]==j)
      if(aj == 0){
        Beta[,j,tt] <- mvrnorm(1, m, V_beta * sigma2[j, tt-1])
        sigma2[j, tt] <- rinvgamma(1,a,b)
        
      }else{
        temp1 = X[which(Z[,tt] == j), , drop=F]
        temp1t= t(temp1)
        temp2 = Y[which(Z[,tt]==j)]
        solve_v_star = temp3 + temp1t %*% temp1
        temptt = solve(solve_v_star)
        
        mu_star = temptt %*% (temp3 %*% m + temp1t %*% temp2)
        
        v_star = temptt
        
        a_star = a + aj/2
        
        b_star = b + 0.5 * (t(m) %*% temp3 %*% m  + t(temp2) %*% temp2 - t(mu_star) %*% solve_v_star %*% mu_star)
        
        
        Beta[,j,tt] <- mvrnorm(1, mu_star, v_star * sigma2[j, tt-1])
        sigma2[j, tt] <- rinvgamma(1,a_star, b_star)  #1/rgamma(1, a_star, b_star)
        
      }
    }
    #   Beta[,1:length(sig_star),tt] <- B_star
    #   sigma2[,tt] <-1
    #   sigma2[1:length(sig_star),tt] <- sig_star
  }
  print(Sys.time()-strt)
}

source('Analysis.R')
