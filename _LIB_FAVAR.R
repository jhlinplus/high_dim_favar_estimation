## ***************************************************
## Description: FAVAR estimation
##
## - Calibration equation estimation
## - FAVAR estimation
##
## Author: Jiahe Lin
## ***************************************************

source("_LIB_Regularized_VARd_LS.R");
source("_LIB_IC_Calc.R");

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# factor and factor loading estimation based on given Theta
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
factor_extract = function(Theta,rk,IR)
{
    #{param,matrix} Theta: matrix based on which factors and loadings are extracted
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    
    #{rtype,list} the return list has two components:
    #   est_f: estimated factors
    #   est_Lambda: estimated loadings
    Theta_SVD = svd(Theta);
    if ( IR == 'PC3' )
    {
        est_f = Theta[,1:r];
        if (rk == 1)
        {
            est_f = matrix(est_f,ncol=1);
            est_Lambda0 = matrix(Theta_SVD$v[,1],ncol=1) * Theta_SVD$d[1]/sqrt(n);
            est_Lambda0 = est_Lambda0 / est_Lambda0[1,1];
        }
        else{
            est_Lambda0 = Theta_SVD$v[,1:rk] %*% diag(Theta_SVD$d[1:rk])/sqrt(n);
            est_Lambda = est_Lambda0 %*% solve(est_Lambda0[1:rk,1:rk]);
        }
    }
    else if ( IR == 'PC1')
    {
        if ( rk == 1 )
        {
            est_f = matrix( sqrt(n) * Theta_SVD$u[,1], ncol=1 );
            est_Lambda = matrix(Theta_SVD$v[,1],ncol=1) * Theta_SVD$d[1]/sqrt(n);
        }
        else
        {
            est_f = sqrt(nrow(Theta)) * Theta_SVD$u[,1:rk];
            est_Lambda = Theta_SVD$v[,1:rk] %*% diag(Theta_SVD$d[1:rk])/sqrt(n);
        }
    }
    else if ( IR == 'PC2' )
    {
        if ( rk > 1 )
        {
            est_f0 = sqrt(n) * Theta_SVD$u[,rk];
            est_Lambda0 = matrix(Theta_SVD$v[,1:rk]) * Theta_SVD$d[1:rk]/sqrt(n);
            Q = qr.Q(qr(est_Lambda0[1:rk,1:rk]));
            est_f = est_f %*% Q;
            est_Lambda = est_Lambda0 %*% Q;
        }
        else
        {
            est_f = matrix( sqrt(n) * Theta_SVD$u[,1:rk], ncol=1 );
            est_Lambda = Theta_SVD$v[,1:rk] %*% diag(Theta_SVD$d[1:rk])/sqrt(n);
        }
    }
    else
        stop("Unrecognized IR specification in factor_extract().\n");
    
    return(list(est_f=est_f,est_Lambda=est_Lambda));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Estimate the information eqn based on given r and lambda
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
info_est = function(Y,X,rk,lambda,IR='PC3',parallel=FALSE,verbose=FALSE)
{
    #{param,matrix} Y: observed reseponse matrix data
    #{param,matrix} X: observed covariate matrix data
    #{param,double} rk: rank constraint for the optimization problem
    #{param,double} lambda: penalty parameter for the coef matrix of the observed covariate
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   est_Theta: estimated factor hyperplane
    #   est_B: estimated coefficient matrix for the observed covariates
    #   est_f: estimated factor subject to IR
    #   est_Lambda: estimated factor loadings subject to IR
    
    n = nrow(X);
    dimX = ncol(X);
    dimY = ncol(Y);
    
    ## initialize with Theta = 0
    Theta = array(0,c(n,dimY));
    ## allocate memory for matrix B
    B = array(0,c(dimY,dimX));
    
    ###################
    ## routine for calculation objective function
    ##
    ## ||Y - XB ||_F^2/(2n) + lambda||B||_1
    ####################
    obj_val = function(Y,X,B,Theta,lambda)
    {
        n = nrow(Y);
        loss = norm(as.matrix(Y - X %*% t(B) - Theta),'F')^2/(2*nrow(Y));
        penalty = lambda * sum(abs(B));
        return(loss + penalty)
    }

    iter = 0;
    fval0 = obj_val(Y,X,B,Theta,lambda);
    fval = c();
    
    CONVERGE = FALSE;
    while( !CONVERGE )
    {
        iter = iter + 1;
        # update B (dimY * dimX) by Lasso
        if (parallel)
        {
            Brow_list = foreach(j = 1:dimY,.packaves=pkgs) %dopar %
            {
                temp = glmnet(x=X,y=(Y-Theta)[,j],lambda=lambda,intercept=FALSE);
                Bj = as.numeric(temp$beta);
            }
            for (j in 1:dimY)
                B[j,] = Brow_list[[j]];
        }
        else
        {
            for (j in 1:dimY)
            {
                temp = glmnet(x=X,y=(Y-Theta)[,j],lambda=lambda,intercept=FALSE);
                B[j,] = as.numeric(temp$beta);
            }
        }
            
        # calcualte the objective function update
        fB =  obj_val(Y,X,B,Theta,lambda);
        fB_update = ifelse(iter==1,fB-fval0,fB-fval[iter-1]);
        
        # update Theta with SVT
        Theta_SVD = svd( Y - X %*%t(B) );
        Theta = Theta_SVD$u %*% diag(c(Theta_SVD$d[1:rk],rep(0,length(Theta_SVD$d)-rk))) %*% t(Theta_SVD$v);
        
        # calcualte the objective function update
        fval[iter] = obj_val(Y,X,B,Theta,lambda);
        fTheta_update = fval[iter] - fB;
        
        f_update = fB_update + fTheta_update;
        
        if (verbose)
        {
            cat(sprintf(">> iter = %d, fB_update = %.4f, fTheta_update = %.4f, f_update = %.4f.\n", iter, fB_update, fTheta_update, f_update));
        }
        CONVERGE = abs(f_update) < 1e-4;
        
        if (iter > 5000)
            stop("Iteration overflow @info_est().\n")
    }
    
    est_fLambda = factor_extract(Theta,rk=rk,IR=IR);
    return(list(est_Theta = Theta,
                est_B = B,
                est_f = est_fLambda$est_f,
                est_Lambda = est_fLambda$est_Lambda));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Automatic estimate the information eqn over a lattice of lambda nand
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
info_auto = function(Y,X,rk_seq,lambda_seq,IR='PC3',parallel=FALSE,verbose=FALSE)
{
    #{param,matrix} Y: observed reseponse matrix data
    #{param,matrix} X: observed covariate matrix data
    #{param,vector} rk_seq: sequence of rank constraints for the optimization problem
    #{param,vector} lambda_seq: sequence of penalty parameter for the coef matrix of the observed covariate
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   out: output for the information equation estimation
    #   idx: the index of the active rank/lambda in the lattice
    #   rk_active: the active rank selected
    #   lambda_active: the active tuning parameter selected
    
    n = nrow(X);
    dimX = ncol(X);
    dimY = ncol(Y);
    
    if (is.null(rk_seq))
    {
        rk_seq = seq(from=1,to=10,by=1);
        cat(sprintf("rk_seq is NULL, set to seq(from=1,to=10,by=1).\n"));
    }
        
    if (is.null(lambda_seq))
    {
        lambda_seq = seq(from=0.1,to=1,by=0.1)*sqrt(log(dimX)/n);
        cat(sprintf("lambda_seq is NULL, set to seq(from=0.1,to=10,by=0.1)*sqrt(logp/n).\n"));
    }
    
    
    PIC_mtx = array(0,c(length(rk_seq),length(lambda_seq)));
    for (j1 in 1:length(rk_seq))
    {
        for (j2 in 1:length(lambda_seq))
        {
            if (verbose)
            {
                cat(sprintf(">> j1 = %d/%d, rk = %d, j2 = %d/%d.\n",
                    j1, length(rk_seq), rk_seq[j1], j2, length(lambda_seq)));
            }
            out = info_est(Y=Y,X=X,rk=rk_seq[j1],lambda=lambda_seq[j2],IR=IR,parallel=parallel,verbose=FALSE);
            
            PIC_mtx[j1,j2] = PIC_calc(Y,X,out$est_B,out$est_Theta,r=rk_seq[j1])$PIC;
        }
    }
    idx = which(PIC_mtx == min(PIC_mtx),arr.ind=TRUE)[1,];
    
    rk_active = rk_seq[idx[1]];
    lambda_active = lambda_seq[idx[2]];
    
    if (verbose)
    {
        cat("Done with selection, proceed with the final information eqn estimation.\n");
    }
    
    out = info_est(Y,X,rk=rk_active,lambda=lambda_active,IR=IR,parallel=parallel,verbose=verbose);
    
    return(list(out=out,idx=idx,rk_active=rk_active,lambda_active=lambda_active));
 
}
    
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Estimate the FAVAR model based on given rank/penalty parameters
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
favar_est = function(Y,X,rk,lambda_Gamma,IR='PC3',lambda_A,d=1,alpha=1,parallel=TRUE,verbose=FALSE)
{
    #{param,matrix} Y: observed reseponse matrix data
    #{param,matrix} X: observed covariate matrix data
    #{param,double} rk: rank constraint for the optimization problem
    #{param,double} lambda_Gamma: penalty parameter for the coef matrix of the observed covariate in the calibration equation
    #{param,string} IR: identification restriction type for extracting factors
    #   'PC1': factors are assumed orthogonal, Lambda'Lamdba diagonal;
    #   'PC2': factors are assumed orthogonal, Lambda is lower triangular;
    #   'PC3': factors are unrestricted
    #   see also Bai & NG, 2013, J of Econometrics
    #{param,double} lambda_A: penalty parameter for transition matrix in the VAR equation estimation
    #{param,d}: lag of the VAR process to estimate
    #{param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
    
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   est_Gamma: estimated coefficient matrix for the observed covariates in the calibration equation
    #   est_f: estimated factor subject to IR
    #   est_Lambda: estimated factor loadings subject to IR
    #   est_A: a list with each component corresponding to the transition matrix estimate
    
    n = nrow(X);
    dimX = ncol(X);
    dimY = ncol(Y);
    
    out_info = info_est(Y=Y,X=X,rk=rk,lambda=lambda_Gamma,IR=IR,parallel=parallel,verbose=verbose);
    
    # joint process (F_t,X_t)
    FX = cbind(out_info$est_f,X);
    # estimate the joint VAR process
    out_VAR = regularized_VAR_est (Y=NULL,X=FX,d=d,lambda=lambda_A,penalty.factor=NULL,alpha=alpha,refit=FALSE,parallel=parallel);
    
    return(list(est_Gamma = info_out$est_B,
                est_Lambda = info_out$est_Lambda,
                est_f = info_out$est_f,
                est_A = out_VAR$Alist)
        )
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Automatic sparse FAVAR estimation
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
favar_auto = function(Y,X,rk_seq,lambda_Gamma_seq,IR='PC3',lambda_A_seq,d=1,alpha=1,parallel=FALSE,verbose=FALSE)
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
    #{param,d}: lag of the VAR process to estimate
    #{param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
    
    #{param,boolean} parallel: whether run parallel when doing Lasso regressions;
    #{param,boolean} verbose: whether print out tracker for each iteration;
    
    #{rtype,list} the return list has the following components:
    #   est_Gamma: estimated coefficient matrix for the observed covariates in the calibration equation
    #   est_f: estimated factor subject to IR
    #   est_Lambda: estimated factor loadings subject to IR
    #   est_A: a list with each component corresponding to the transition matrix estimate
    
    n = nrow(X);
    dimX = ncol(X);
    dimY = ncol(Y);
    
    cat("==== Stage I: estimating calibration eqn with auto-selected rank and penalty params ====\n");
    
    out_autoinfo = info_auto function(Y=Y,X=Y,rk_seq=rk_seq,lambda_seq=lambda_Gamma_seq,IR=IR,parallel=parallel,verbose=FALSE);
    
    out_info = out_autoinfo$out;
    
    # joint process (F_t,X_t)
    FX = cbind(out_info$est_f,X);
    
    cat("==== Stage II: estimating VAR eqn with auto-selected penalty params ====\n");
    
    # auto estimate the joint VAR process
    out_autoVAR = regularized_VAR_auto = function(Y=NULL,X=FX,d=d,alpha=1,lambda_seq=lambda_A_seq,selection="bic",refit=FALSE,parallel=TRUE);
    
    return(list(est_Gamma = info_out$est_B,
                est_Lambda = info_out$est_Lambda,
                est_f = info_out$est_f,
                est_A = out_autoVAR$Alist))
}
