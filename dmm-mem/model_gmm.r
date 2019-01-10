strt<-Sys.time()
library(MASS)
library(mvtnorm)
library(MCMCpack)
library(Rcpp)
sourceCpp("express.cc")

dot = read.csv("/Users/Henryguo/Desktop/code_GMM/Archive/fn.csv", header=TRUE)
set.seed(6781)
ss <- which(dot[,2] == 0)
ss1 <- which(dot[,2] == 1)
Y = as.matrix(dot[ss,1])

n <- nrow(Y)
X <- as.matrix(dot[ss,4:6])

for(iter in 1:10){
  print (iter)
  ssel <- sample(n, length(ss1))
  Ytest <- Y[ssel]
  Xtest <- X[ssel,]
  Xtest <- cbind(rep(1,length(ss1)), Xtest)
  tmp <- 1:n; tmp <- tmp[-ssel]
  Y <- Y[tmp]
  X <- X[tmp,]
  n <- nrow(X)
  X <- cbind(rep(1,n), X)
  p <- ncol(X)
  
  
  Xtest1 <- as.matrix(dot[ss1,4:6])
  Ytest1 <- as.matrix(dot[ss1,1])
  Xtest1 <- cbind(rep(1, length(ss1)), Xtest1)
  
  ### (beta, sigma2) ~ NIG(m, V_beta, a,b)
  V_beta <- diag(3,p)
  m <- rep(0,p)#colMeans(X)#rep(0,p)#colMeans(X)
  J = 15;
  e = 5;
  f = 1; #alpha
  TT = 20000;
  b = 1
  a = 1
  temp3 = solve(V_beta)
  
  ######Initialize parameters for MCMC
  alpha <- rep(0, TT); alpha[1] <- 5
  nu <- array(0, dim = c(J-1, TT))
  nu[,1] <- rbeta(J-1,1,alpha[1])
  pp <- array(0, dim = c(J, TT))
  pp[1,1] <- nu[1,1]
  tmp <- 1-nu[1,1]
  for(jj in 2:(J-1)){
    pp[jj,1] = tmp * nu[jj,1]
    tmp <- tmp * (1-nu[jj,1])
  }
  pp[J,1] <- tmp
  Beta <- array(0, dim = c(p,J,TT))
  sigma2 <- array(0, dim=c(J,TT))
  for(j in 1:J){
    sigma2[j, 1] <- rinvgamma(1,a,b) #1 /rgamma(1, a, b)
    Beta[, j, 1] <- mvrnorm(1, m, V_beta * sigma2[j, 1])
  }
  
  
  #Z = rep(0, n)
  Z <- array(0,dim=c(n,TT))
  for (tt in 2 : TT){
    print (tt)
    helpa = log(pp[,tt-1])
    helpb = t(Beta[,,tt-1])
    helpc = sqrt(sigma2[, tt-1])
    
    Z[,tt] = .loopexp2(helpa, helpb, helpc, X, Y, J, n)
    
    #Z <- Z_true
    ### update nu
    for(j in 1:(J-1)){
      aj <- sum(Z[,tt] == j)
      as <- sum(Z[,tt] >j)
      nu[j,tt] <- rbeta(1, aj + 1, alpha[tt-1] + as)
    }
    
    pp[1,tt] <- nu[1,tt]
    tmp <- 1-nu[1,tt]
    for(jj in 2:(J-1)){
      pp[jj,tt] = tmp * nu[jj,tt]
      tmp <- tmp * (1-nu[jj,tt])
    }
    pp[J,tt] <- tmp
    
    # pp[1:length(pp_star),tt] <- pp_star
    
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
        sigma2[j, tt] <- rinvgamma(1,a_star, b_star)#1/rgamma(1, a_star, b_star)
        
      }
    }
    #   Beta[,1:length(sig_star),tt] <- B_star
    #   sigma2[,tt] <-1
    #   sigma2[1:length(sig_star),tt] <- sig_star
  }
  cat("iter = ", iter, "\n")
#source("/Users/Henryguo/Desktop/code_GMM/Archive/Analysis.R")
#DF <- data.frame("Yhat" = Yhat, "Ytest" = Ytest, "Xtest" = Xtest[,2:4], "flag" = "0")
#tmp <- data.frame("Yhat" = Yhat1, "Ytest" = Ytest1, "Xtest" = Xtest1[,2:4], "flag" = "1")
#DF <- rbind(DF, tmp)
#save.image(file =paste0("/gpfs/home/lul37/code/result_", iter, ".RData", sep = ""))
}
print(Sys.time()-strt)
