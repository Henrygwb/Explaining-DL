set.seed(1567)
setwd("/home/wzg13/Work/ElasticNet_GMM/explainability/DMMEN/fashion")
library(MASS)
library(mvtnorm)
library(MCMCpack)
library(Rcpp)
library(R.matlab)
library(mvtnorm)
library(statmod)
library(truncnorm)
sourceCpp("express.cc")
a_lambda <- function(lambda_1, lambda_2, lambda_1_er, lambda_2_er, tau, sigma2){
  a_1 <- 1
  idx <- which(c_j[,tt] == k)
  for (j in idx){
    prob_earlier <- prob_tau(tau[, j], sigma2[j], lambda_1_er, lambda_2_er) 
    prob_later <- prob_tau(tau[, j], sigma2[j], lambda_1, lambda_2) 
    tmp <- prob_later/prob_earlier
    a_1 <- a_1 * tmp
  }
  return(a_1)
}

a_sigma2 <- function(sigma2_tt_1, sigma2_tt, tau, lambda_1, lambda_2){
  prob_earlier <- prob_tau(tau, sigma2_tt_1, lambda_1, lambda_2)  
  prob_later <- prob_tau(tau, sigma2_tt, lambda_1, lambda_2)
  a_1 <- prob_later/prob_earlier
  return(a_1)
}

prob_tau <- function(tau, sigma2, lambda_1, lambda_2){
  tmp_1 <- pnorm(-lambda_1/sqrt(4*sigma2*lambda_2))
  tmp_1 <- tmp_1^(-p)
  tmp_2 <- (lambda_1^2/(8*pi*sigma2*lambda_2))^(p/2)
  tmp_3 <- -(lambda_1^2/(8*sigma2*lambda_2))*(sum(tau^-1)) 
  tmp_3 <- exp(tmp_3)
  tmp_4 <- prod(tau^(-1.5))
  prob <- (2^(-p))*(tmp_1)*(tmp_2)*(tmp_3)*(tmp_4)
  return(prob)
}

print ('Loading data...')

for (na in 1:10){
  ptm <- Sys.time()
  na = na-1
  name = paste('data_for_explain_', na, sep='')
  name = paste(name, 'mat', sep='.')
  print (name)
  data <- readMat(name)
  X_original <- data[['data']]
  Y_original <- data[['pred']]
  segment <- data[['seg']]
  #n <- nrow(Y_original)
  #ss1 <- as.integer(n*1)
  X <- X_original
  Y <- Y_original
  print ('Training...')
  n <- nrow(X)
  p <- ncol(X)
  print (dim(Y))
  print (n)
  print (p)
  
  #Seting hyperparameters for prior##########################
  ###### number of iterations
  TT = 20000;
  ###### number of Gaussian componenets
  J = 6;
  ###### number of elastic nets
  K = 3;
  ###### alpha ~ Gamma(e, f)
  e = 5;
  f = 1; 
  ###### alpha_e = 1/K
  alpha_e = 1/K
  ###### sigma2 ~ InGamma(a, b)
  b = 1/2;
  a = 1/2;
  ###### lambda_k_1 ~ Gamma(Lk, vk/2), lambda_1_2 ~ Gamma(Lk, vk/2)
  L = 2;
  R = 1.5;
  v = 2;
  
  #Initialize parameters for Gibbs sampling##################
  ###### alpha ~ Gamma(e, f)
  alpha <- rep(0, TT); alpha[1] <- e
  
  ###### mu ~ Beta(1, alpha)
  mu <- array(0, dim = c(J-1, TT))
  mu[,1] <- rbeta(J-1, 1, alpha[1])
  
  ###### \pi_j = \mu_j\prod_{l=1}^{j-1}(1-\mu_l)
  Pi <- array(0, dim = c(J, TT))
  Pi[1,1] <- mu[1,1]
  tmp <- 1-mu[1,1]
  for(jj in 2:(J-1)){
    Pi[jj,1] = tmp * mu[jj,1]
    tmp <- tmp * (1-mu[jj,1])
  }
  Pi[J,1] <- tmp
  
  ###### Z
  Z <- array(0, dim = c(n,TT)) 
  
  ###### alpha_e = 1/K
  
  ###### w ~ Diri(alpha_e)
  w <- array(0, dim = c(K, TT)) 
  w[, 1] <- rdirichlet(1,rep(alpha_e, K))
  
  
  ###### c_j
  c_j <- array(0, dim = c(J,TT))
  c_j[,1] <- sample(K, J, replace = TRUE, prob = w[,1])
  ###### lambda
  lambda <- array(0, dim = c(K, 2, TT))
  
  for (k in 1:K){
    ###### lambda_k <- lambda[k,:] lambda_k_1 ~ Gamma(Lk, vk/2), lambda_k_2 ~ Gamma(Rk, vk/2)
    lambda[k, 1, 1] <- rgamma(1, L, v/2)
    lambda[k, 2, 1] <- rgamma(1, R, v/2)
  }
  
  
  ###### sigma2 ~ InGamma(a, b)
  sigma2 <- array(0, dim=c(J,TT))
  for(j in 1:J){
    sigma2[j, 1] <- rinvgamma(1,a,b) #1 /rgamma(1, a, b)
  }
  
  ###### tau and Beta  Beta ~ N(0, (sigma^2/lambda_2)diag(1-tau)), tau ~ IG(0.5, 0.5(lambda_1^2/4*simga2*lambda_2))
  tau <- array(0, dim=c(p, J, TT))
  Beta <- array(0, dim = c(p, J, TT))
  for (j in 1:J){
    cc = c_j[j, 1] 
    ################################## tau ~ rinvgamma_{(0,1)}(0.5, 0.5(tmp))
    #tmp <- 0.5 * ((lambda[cc, 1, 1]^2)/(4*lambda[cc, 2, 1]*sigma2[j, 1]))
    #tau[ , j, 1] <- rinvgamma(p, 0.5, tmp)
    tau[ , j, 1] <- rep(0.5, p)
    S_tau = diag((1-tau[ , j, 1]))
    Beta[,j,1] <- mvrnorm(1, rep(0, p),  Sigma = ((sigma2[j, 1]/lambda[cc, 2, 1]) * S_tau))
  }
  
  #Gibbs sampling##################
  for (tt in 2 : TT){
    #print (tt)
    #strt <- Sys.time()
    ### 1. update Z
    helpa = log(Pi[,tt-1])
    helpb = t(Beta[,,tt-1])
    helpc = sqrt(sigma2[, tt-1])  
    Z[,tt] = .loopexp2(helpa, helpb, helpc, X, Y, J, n)
    
    ### 2. update mu
    for(j in 1:(J-1)){
      aj <- sum(Z[,tt] == j)
      as <- sum(Z[,tt] >j)
      mu[j,tt] <- rbeta(1, aj + 1, alpha[tt-1] + as)
    }
    
    ### 3. update Pi
    Pi[1,tt] <- mu[1,tt]
    tmp <- 1-mu[1,tt]
    for(jj in 2:(J-1)){
      Pi[jj,tt] = tmp * mu[jj,tt]
      tmp <- tmp * (1-mu[jj,tt])
    }
    Pi[J,tt] <- tmp
    
    ### 4. update alpha
    alpha[tt] <- rgamma(1,J+e-1, rate = f- sum(log(1-mu[,tt])))
    
    ### 5. update c_j
    prob_cj <- rep(0, K)
    for (j in 1:J){
      for (k in 1:K){
        mean <- rep(0, p) 
        S_tau <- diag((1-tau[, j, tt-1]))
        tmp <- sigma2[j, tt-1]/lambda[k, 2, tt-1] * S_tau 
        prob_cj[k] <- dmvnorm(Beta[, j, tt-1], mean, tmp, log = TRUE) + log(w[k, tt-1]) 
      }
      mmax <- max(prob_cj)
      deno <- mmax+log(sum(exp(prob_cj - mmax)))
      prob_cj <- exp(prob_cj - deno)
      c_j[j, tt] <- sample(K, 1, prob = prob_cj)
    }
    
    ### 6. update w
    idx <- rep(0, K)
    for (k in 1:K){
      idx[k] <- sum(c_j[,tt] == k)
    }
    w[, tt] <- rdirichlet(1, alpha = (alpha_e + idx))
    
    ### 7. update alpha_e = 1/K
    
    ### 8. update mu, tau and Sigma
    for(j in 1:J){
      aj <- sum(Z[,tt]==j)
      cc = c_j[j, tt] 
      S_tau <- diag((1-tau[, j, tt-1]))
      if(aj == 0){
        ################################## tau ~ rinvgamma_{(0,1)}(0.5, 0.5(tmp))
        #tmp <- 0.5 * ((lambda[cc, 1, tt-1]^2)/(4*lambda[cc, 2, tt-1]*sigma2[j, tt-1]))
        #tau[ , j, tt] <- rinvgamma(p, 0.5, tmp)
        tau[ , j, tt] <- rep(0.5, p)
        Beta[,j,1] <- mvrnorm(1, rep(0, p),  Sigma = ((sigma2[j, tt-1]/lambda[cc, 2, tt-1]) * S_tau))
        sigma2[j, tt] <- rinvgamma(1, a, b)       
      }else{
        temp_x <- X[which(Z[,tt] == j), , drop=F]
        temp_xt <- t(temp_x)
        temp_y <- Y[which(Z[,tt]==j)]
        
        ### 8.1 update mu
        inv_S_tau <- solve(S_tau)
        inv_R_tau <- temp_xt %*% temp_x + lambda[cc,2,tt-1] * inv_S_tau
        R_tau <- solve(inv_R_tau)
        
        mean_star <- R_tau %*% temp_xt %*% temp_y
        co_star <- R_tau * sigma2[j, tt-1]
        Beta[,j,tt] <- mvrnorm(1, mean_star,  co_star)
        
        ### 8.2 update tau
        for (pp in 1:p){
          dj_temp <- lambda[cc , 1, tt-1]/(2*lambda[cc , 2, tt-1]*abs(Beta[pp, j, tt-1]))  
          c_temp <- (lambda[cc , 1, tt-1]^2)/(4*lambda[cc , 2, tt-1]*sigma2[j, tt-1])
          ll_tmpe <- rinvgauss(1, mean = dj_temp, shape = c_temp)
          tau[pp, j, tt] <- ll_tmpe/(1 + ll_tmpe)
        }
        
        ### 8.3 update sigma2 (Metroplis-Hastings Sampling)
        ##sample new sigma2 from inverse gamma
        a_star <- a + (aj + 1)/2
        tmp_b_1 <- sum((temp_y - temp_x %*% Beta[, j, tt-1])^2)
        tmp_b_2 <- lambda[cc , 2, tt-1]*(t(Beta[, j, tt-1]) %*% inv_S_tau %*% Beta[, j, tt-1])
        b_star <- b + 0.5*(tmp_b_1 + tmp_b_2) 
        sigma2_tt <- rinvgamma(1, a_star, b_star)#1/rgamma(1, a_star, b_star)
        ##decide whether accepts the new sigma2 or not
        u <- runif(1, min=0, max=1)
        ## compute p(tau|sigma2(t))/p(tau|sigma2(t-1))
        a_s <- a_sigma2(sigma2[j, tt-1], sigma2_tt, tau[, j, tt-1], lambda[cc , 1, tt-1], lambda[cc , 2, tt-1])
        if (is.nan(a_s)){
          a_s = 0
        }
        if (u < min(a_s, 1)){
          #print ('accept')
          sigma2[j, tt] <- sigma2_tt 
        }
        else{
          sigma2[j, tt] <- sigma2[j, tt-1]
        }
      }
    }
    
    ### 9. update lambda 
    for (k in 1:K){
      ## sample lambda from prosprol distribution
      for (ii in 1:100){
        lambda_1 <- rtruncnorm(1, mean = lambda[k, 1, tt-1], sd = sqrt(2))
        if (lambda_1 > 0){
          break
        }    
      }
      for (ii in 1:100){
        lambda_2 <- rtruncnorm(1, mean = lambda[k, 2, tt-1], sd = sqrt(2))
        if (lambda_2 > 0){
          break
        }    
      }
      u <- runif(1, min=0, max=1)
      a_l <- a_lambda(lambda_1, lambda_2, lambda[k, 1, tt-1], lambda[k, 2, tt-1], tau[, , tt-1], sigma2[ ,tt-1])
      if (is.nan(a_l)){
        a_l = 0
      }
      if (u < min(a_l, 1)){
        lambda[k, 1, tt] <- lambda_1
        lambda[k, 2, tt] <- lambda_2
      }
      else{
        lambda[k, 1, tt] <- lambda[k, 1, tt-1]
        lambda[k, 2, tt] <- lambda[k, 2, tt-1]
      }
    }
    if (tt%%100==0){
      print (tt)
    }	
    #print (Sys.time()-ptm)
  }
  print (Sys.time()-ptm)
  #source('Analysis_0.R') 
  #source('final_params.R')
}


