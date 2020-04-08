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
rk_seq = seq(from=3,to=7,by=1) ## tuning parameter sequence for the rank specification.
lambda_Gamma_seq = seq(from=0.25,to=3,by=0.25)*sqrt(log(ncol(X))/nrow(X)); ## tuning parameter sequence for the penalty corresponding to Gamma, can be NULL
lambda_A_seq = NULL ## tuning parameter for the regularized estimation for the VAR process, can be NULL (read the comments to see the default setup)

### Other specifications: 
#>> factors are recovered using PC3, i.e., no restrictions
#>> the VAR process is estimated using lasso (alpha=1), with the true lag specification (d=2)
IR = "PC3";
d = 2;
alpha = 1;

## run FAVAR, tuning parameters are automatically selected based on a lattice of tuning parameters
out_autoFAVAR = favar_auto(Y,X,rk_seq,lambda_Gamma_seq,IR=IR,lambda_A_seq,d=d,alpha=alpha,parallel=TRUE,verbose=TRUE)

################################
## extracting estimated components
################################
## estimated Gamma -- regression coefficient in the calibration (information) equation
out_autoFAVAR$est_Gamma

## estimated hyperplane, i.e., Theta-hat
out_autoFAVAR$est_Theta

## estimated leading transition matrix (lag=1) for the VAR process
out_autoFAVAR$est_A[[1]]

## estimated transition matrix for lag = 2 for the VAR process
out_autoFAVAR$est_A[[2]]
