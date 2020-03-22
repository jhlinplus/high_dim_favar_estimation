## ***************************************************************************
## Description: Example for using the FAVAR estimation procedure
##
## Sample size: 100
## Number of factors: 5
## Dimension of Y: 100
## Dimension of X: 100
## The joint process of F and X are generated according to a VAR(2) process
## where the spectral radius of the transition matrix for the stacked process is 0.8
## Note that in the DGP, the factors are unrestricted (i.e., no orthogonal)
## 
## ***************************************************************************

setwd("~/GitHub/High_dim_FAVAR_estimation")

rm(list=ls())
source("_LIB_FAVAR.R");
source("_LIB_Eval.R");

## Read in data
Y = read.csv("example_data/Y.csv")[,-1]; ## first column records the time index, hence removed
X = read.csv("example_data/X.csv")[,-1]; ## first column records the time index, hence removed

################################
## estimate using auto FAVAR 
################################
### tuning parameter specification
rk_seq = seq(from=3,to=7,by=1) ## can be NULL
lambda_Gamma_seq = seq(from=0.25,to=3,by=0.25)*sqrt(log(ncol(X))/nrow(X)); ## can be NULL
lambda_A_seq = NULL
### Other specifications: 
#>> factors are recovered using PC3, i.e., no restrictions
#>> the VAR process is estimated using lasso (alpha=1), with the true lag specification (d=2)
IR = "PC3";
d = 2;
alpha = 1;
## run
out_autoFAVAR = favar_auto(Y,X,rk_seq,lambda_Gamma_seq,IR=IR,lambda_A_seq,d=d,alpha=alpha,parallel=TRUE,verbose=TRUE)

################################
## evaluate the performance -- note that this is only valid for synthetic data
################################

## estimated Gamma
trueGamma = read.csv("example_data/trueGamma.csv");
Eval_Sparse(trueGamma, out_autoFAVAR$est_Gamma)

## estimated hyperplane
trueF = as.matrix(read.csv("example_data/trueF.csv")[,-1]);
trueLambda = as.matrix(read.csv("example_data/trueLambda.csv"));
Eval_Flats(trueF %*% t(trueLambda), out_autoFAVAR$est_Theta)$metrics

## estimated transition matrix
trueA1 = read.csv("example_data/trueA1.csv");
Eval_Sparse(trueA1, out_autoFAVAR$est_A[[1]])

trueA2 = read.csv("example_data/trueA1.csv");
Eval_Sparse(trueA2, out_autoFAVAR$est_A[[2]])
