#include <RcppArmadilloExtensions/sample.h>
#include <iostream>

using namespace Rcpp ;

// [[Rcpp::depends(RcppArmadillo)]]


double find_max(Rcpp::NumericVector& z, int J){
  double t = z[0];
  int i;
  for (i = 1; i < J; i++)
  {
    if (z[i] > t)
      t = z[i];
  }
  return t;
}

double find_max_arma(arma::rowvec& z, int J){
  double t = z[0];
  int i;
  for (i = 1; i < J; i++)
  {
    if (z[i] > t)
      t = z[i];
  }
  return t;
}

// [[Rcpp::export(".loopexp1")]]
void loopexp_out1(const Rcpp::NumericVector& helpa, const arma::mat& helpb, const arma::vec& helpc,
                  const arma::mat& X, const arma::vec& Y, int J, int n, Rcpp::NumericVector& Z)
{
  int i, j;
  
  for (i = 1; i <= n; i++)
  {
    double t1 = Y(i-1);
    arma::rowvec t2 = X.row(i-1);
    Rcpp::NumericVector prob(J);
    
    for (j = 1; j <= J; j++)
    {
      arma::mat t = (helpb.row(j-1)) * (t2).t();
      double tt = t(0,0);
      double dnor = Rf_dnorm4(t1, tt, helpc[j-1], 1);
      prob[j-1] = helpa[j-1] + dnor;
    }
    
    
    double mmax = find_max(prob, J);
    double deno = mmax + log(sum(exp(prob - mmax)));
    Rcpp::IntegerVector JJ = seq_len(J) ; // creates 1:15 vector
    
    Z[i-1] = RcppArmadillo::sample(JJ, 1, false, (exp(prob - deno)))[0];
  }
  
  return;
}



// [[Rcpp::export(".loopexp2")]]
Rcpp::NumericVector loopexp_out2(const Rcpp::NumericVector& helpa, const arma::mat& helpb, const arma::vec& helpc,
                                 const arma::mat& X, const arma::vec& Y, int J, int n)
{
  int i, j;
  
  Rcpp::NumericVector Z(n);
  for (i = 1; i <= n; i++)
  {
    double t1 = Y(i-1);
    arma::rowvec t2 = X.row(i-1);
    Rcpp::NumericVector prob(J);
    
    for (j = 1; j <= J; j++)
    {
      arma::mat t = (helpb.row(j-1)) * (t2).t();
      double tt = t(0,0);
      double dnor = Rf_dnorm4(t1, tt, helpc[j-1], 1);
      prob[j-1] = helpa[j-1] + dnor;
    }
    
    
    double mmax = find_max(prob, J);
    double deno = mmax + log(sum(exp(prob - mmax)));
    Rcpp::IntegerVector JJ = seq_len(J) ; // creates 1:15 vector
    
    Z[i-1] = RcppArmadillo::sample(JJ, 1, false, (exp(prob - deno)))[0];
  }
  
  return Z;
}

// [[Rcpp::export(".loopexp3")]]
arma::mat loopexp_out3(const Rcpp::NumericVector& helpa, const arma::mat& helpb,
                       const arma::vec& helpc,
                       const arma::mat& X, const arma::vec& Y, int J, int n)
{
  int i, j;
  arma::mat prob(n, J);
  for (i = 1; i <= n; i++)
  {
    double t1 = Y(i-1);
    arma::rowvec t2 = X.row(i-1);
    
    for (j = 1; j <= J; j++)
    {
      arma::mat t = (helpb.row(j-1)) * (t2).t();
      double tt = t(0,0);
      double dnor = Rf_dnorm4(t1, tt, helpc[j-1], 1);
      prob(i-1,j-1) = helpa[j-1] + dnor;
    }
    arma::rowvec test = prob.row(i-1);
    double mmax = find_max_arma(test, J);
    double deno = mmax + log(sum(exp(prob.row(i-1) - mmax)));
    prob.row(i-1) = exp(prob.row(i-1) - deno);
  }
  
  return prob;
  //return ;
}


/*
*
for(ss in 1:ncol(MCMC_miscla$W)){
for(yy in 1:length(Y_star)){
mu_new <- Xnew[ii,]%*%MCMC_miscla$Beta[,,ss]
density[yy] = density[yy] + sum(MCMC_miscla$W[,ss]*
dnorm(Y_star[yy], mu_new, sqrt(MCMC_miscla$Sigma[,ss])))
}
}
density <- density/ncol(MCMC_miscla$W)
*
*/

/*
*
*input :
*        MCMC_miscal_w; Y_star; MCMC_miscla_beta; Xnew;
*
*Output : density
*
*/

NumericVector my_dnorm(const double& x, const NumericVector& means, const NumericVector& sds){
  int n = means.size();
  
  //std::cout << "asdsad1 : " << n << std::endl;
  NumericVector res(n) ;
  
  for( int i=0; i<n; i++)
    res[i] = Rf_dnorm4(x, means[i], sds[i], 0);
  
  //std::cout << "asdsad2\n";
  //    for (int i=0; i < n; i++)
  //        std::cout << res[i] << " ";
  
  return res;
}


NumericVector my_mul(const arma::vec& a, const NumericVector& b){
  int n = a.size();
  int i;
  
  NumericVector res(n);
  
  for(i=0; i<n; i++)
    res[i] = a[i] * b[i];
  
  return res;
}


// [[Rcpp::export(".loopexp4")]]
Rcpp::NumericVector loopexp_out4(const Rcpp::NumericVector& Y_star, const arma::mat& w,
                                 const arma::mat& Xnew, Rcpp::NumericVector beta,
                                 const arma::mat& sigma, int ii)
{
  
  int n_mcmc_w = w.n_cols;
  int y_star_len = Y_star.length();
  NumericVector density(y_star_len);
  int ss, yy;
  
  Rcpp::IntegerVector x_dims = beta.attr("dim");
  arma::cube beta_cube(beta.begin(), x_dims[0], x_dims[1], x_dims[2], false);
  
  
  for (ss = 1; ss <= n_mcmc_w; ss++)
  {
    for (yy = 1; yy <= y_star_len; yy++)
    {
      arma::mat mu_new = Xnew.row(ii-1) * beta_cube.slice(ss-1);
      arma::rowvec mu_new_row = mu_new.row(0);
      Rcpp::NumericVector mu_new1(mu_new_row.begin(), mu_new_row.end());
      arma::vec sigma_sqrt = sqrt(sigma.col(ss-1));
      Rcpp::NumericVector sigma_sqrt_vec(sigma_sqrt.begin(), sigma_sqrt.end());
      
      Rcpp::NumericVector temp = my_dnorm(Y_star[yy-1], mu_new1, sigma_sqrt_vec);
      double sumt =  sum(my_mul(w.col(ss-1), temp));
      density[yy-1] = density[yy-1] + sumt;
    }
  }
  
  density = density / n_mcmc_w;
  
  return density;
  
}

// [[Rcpp::export]]
arma::mat cube_means(Rcpp::NumericVector vx) {
  
  Rcpp::IntegerVector x_dims = vx.attr("dim");
  arma::cube x(vx.begin(), x_dims[0], x_dims[1], x_dims[2], false);
  
  arma::mat result(x.n_cols, x.n_slices);
  for (unsigned int i = 0; i < x.n_slices; i++) {
    result.col(i) = arma::conv_to<arma::colvec>::from(arma::mean(x.slice(i)));
  }
  
  return result;
}
