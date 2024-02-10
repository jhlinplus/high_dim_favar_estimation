## ***************************************************
## Description: library for high-dim FAVAR estimation
##
## - favar_est(): rank constraint and tuning parameters need to be provided; first estimate the info equation, then using the recovered factor,
##                proceed with VAR(d) (regularized) estimation
## - favar_auto(): first estimate the information equation (the optimal choice of tuning parameters is based on PIC),
##                 then estimate the VAR equation (the optimal choice of the tuning parameter is based on BIC).
##
## Author: Jiahe Lin. jiahelin@umich.edu
## ***************************************************

source("srcR/_estimate_regularized_VARd.R");
source("srcR/_estimate_info_eqn.R")
source("srcR/_calc_information_criteria.R");

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Estimate the FAVAR model based on given rank/penalty parameters
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
favar_est = function(Y,X,rk,lambda_Gamma,IR='PC3',lambda_A,penalty_fac=NULL,d=1,alpha=1,parallel=TRUE,verbose=FALSE)
{
    ## params for the information eqn
    #{param,matrix} Y: observed reseponse matrix data
    #{param,matrix} X: observed covariate matrix data
    #{param,double} rk: rank constraint for the optimization problem
    #{param,double} lambda_Gamma: penalty parameter for the coef matrix of the observed covariate in the calibration equation
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    
    ## params for the VAR eqn
    #{param,double} lambda_A: penalty parameter for transition matrix in the VAR equation estimation
    #{param,double} penalty_fac: penalty factor for transition matrix in the VAR equation estimation
    #{param,double} d: lag of the VAR process to estimate
    #{param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
    
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   est_Gamma: estimated coefficient matrix for the observed covariates in the calibration equation
    #   est_f: estimated factor subject to IR
    #   est_Lambda: estimated factor loadings subject to IR
    #   est_A: a list with each component corresponding to the transition matrix estimate
    
    ## estimating the information equation
    out_info = info_est(Y=Y,X=X,rk=rk,lambda=lambda_Gamma,IR=IR,parallel=parallel,verbose=verbose);
    
    # obtain the joint process (F_t,X_t)
    FX = cbind(out_info$est_f,X);
    # regularized estimation of the joint VAR(d) process
    out_VAR = regularized_VAR_est (Y=NULL,X=FX,d=d,lambda=lambda_A,penalty.factor=penalty_fac,alpha=alpha,refit=FALSE,parallel=parallel);
    
    return(list(est_Gamma = out_info$est_B,
                est_Lambda = out_info$est_Lambda,
                est_f = out_info$est_f,
                est_A = out_VAR$Alist,
                est_Theta = out_info$est_Theta)
        )
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Automatic sparse FAVAR estimation
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
favar_auto = function(Y,X,rk_seq,lambda_Gamma_seq,IR='PC3',lambda_A_seq,penalty_fac=NULL,d=1,alpha=1,parallel=FALSE,verbose=FALSE)
{
    #{param,matrix} Y: observed reseponse matrix data
    #{param,matrix} X: observed covariate matrix data
    #{param,vector} rk_seq: sequence of rank constraint for the optimization problem
    #{param,vector} lambda_Gamma_seq: sequence of penalty parameter for the coef matrix of the observed covariate in the calibration equation
    
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    
    #{param,double} lambda_A_seq: sequence of penalty parameter for transition matrix in the VAR equation estimation
    #{param,double} penalty_fac: penalty factor for transition matrix in the VAR equation estimation
    #{param,d}: lag of the VAR process to estimate
    #{param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
    
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   est_Gamma: estimated coefficient matrix for the observed covariates in the calibration equation
    #   est_f: estimated factor subject to IR
    #   est_Lambda: estimated factor loadings subject to IR
    #   est_A: a list with each component corresponding to the transition matrix estimate
    
    Y = as.matrix(Y);
    X = as.matrix(X);
    
    cat(sprintf("[%s] Stage I: estimating calibration eqn with auto-selected rank and penalty params\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")));
    
    out_autoinfo = info_auto(Y=Y,X=X,rk_seq=rk_seq,lambda_seq=lambda_Gamma_seq,IR=IR,parallel=parallel,verbose=verbose);
    out_info = out_autoinfo$out;
    
    cat(sprintf("[%s] Stage II: estimating VAR eqn with auto-selected penalty params\n",format(Sys.time(), "%Y-%m-%d %H:%M:%S")));
    # joint process (F_t,X_t)
    FX = cbind(out_info$est_f,X);
    # auto estimate the joint VAR process
    out_autoVAR = regularized_VAR_auto(Y=NULL,X=FX,d=d,alpha=1,lambda_seq=lambda_A_seq,penalty.factor=penalty_fac,selection="bic",refit=FALSE,parallel=parallel);
    
    cat(sprintf("[%s] Done with estimation \n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")));
    return(list(est_Gamma = out_info$est_B,
                est_Lambda = out_info$est_Lambda,
                est_f = out_info$est_f,
                est_A = out_autoVAR$out$Alist,
                est_Theta = out_info$est_Theta,
                pic = out_autoinfo$pic,
                bic = out_autoVAR$bic))
};
