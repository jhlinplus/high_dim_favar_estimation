## ***************************************************
## Description: main library for estimation functions
##
## - Sparse VAR estimation
## - Calibration equation estimation
## - FAVAR estimation
## - Ridge VAR estimation
##
## Author: Jiahe Lin
## ***************************************************
pkgs = c("glmnet");
new = pkgs[!(pkgs%in%installed.packages()[,"Package"])]
if (length(new)){
    for (pkg in new)
    {
        install.packages(pkg, dependencies = TRUE)
    }
}
sapply(pkgs, require, character.only = TRUE);

source("_LIB_IC_Calc.R");

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# Estimate of a sparse VAR with the given penalty parameter
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
sparse_VAR_est = function(Y=NULL,X,d=NULL,lambda,penalty.factor=NULL,refit=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
	#	if NULL, then response will be extracted from X
	#	if not NULL, effectively running a multivariate lasso regression Y~X
	# {param,matrix} X: regressor matrix/time series observations stacked in rows
	# {param,double} d: VAR(d) model will be estimated if d is not NULL
	# {param,double} lambda: penalty/tuning parameter for lasso estimate
	# {param,vector} penalty.factor: vector that allows different coordinates to get different penalty
	# {param,boolean} refit: whether do refit after obtaining a low-dim support 
	
	# rtype: a list of two elements A (col-concatenated version), Alist
	
	n = nrow(X); p = dim(X);
    if (is.null(Y))
	{
    	if (is.null(lag))
			stop("Error in Est_VAR(): response matrix is not specified. d must be non NULL.\n");
		
		n_initial = nrow(X);
		Y = X[(d+1):n,];
		Xlist = list();
		for (i in 1:d)
			Xlist[[i]] = X[(d-i+1):(n-i),]
		X = do.call("cbind",Xlist);
    }
	else
	{
		if (!is.null(d))
			warning("Both X and Y are provided; d will not be active.\n");
		d = NULL;
	}

	dimX = ncol(X); dimY = ncol(Y);
	stopifnot( nrow(X) == nrow(Y) );
	
	if (is.null(penalty.factor))
	{
        A = array(0,c(dimY,dimX));
        for (j in 1:dimY)
		{
            temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE);
            A[j,] = as.numeric(temp$beta);
        }
    }
    else
	{
        stopifnot(length(penalty.factor)==dimX);
		A = array(0,c(dimY,dimX));
        for (j in 1:dimY)
		{
            temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE,penalty.factor=penalty.factor);
            A[j,] = as.numeric(temp$beta);
        }
    }
    if (refit)
	{
        skeleton = 1*(A!=0);
        for (j in 1:dimY)
		{
            skeleton_temp = which(skeleton[j,]!=0);
            if ( length(skeleton_temp) < nrow(X) && length(skeleton_temp) > 0 ){
                A[,skeleton_temp] = as.numeric(lm(Y[,j]~X[,skeleton_temp]+0)$coef);
            }
            else
                next;
        }
    }
    if (!is.null(d))
	{
		Alist = lapply(1:d,function(x){A[,(1+(x-1)*p):(x*p)]});
		names(Alist) = paste("lag=",1:d,sep=""); 
	}
	else
		Alist = A;
	
    return(list(A=A,Alist));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# Estimate of a sparse VAR with auto-selected tuning parameters using bic
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
bic_sparse_VAR = function(Y=NULL,X,d=NULL,lambda_seq=NULL,penalty.factor=NULL,refit=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
    #    if NULL, then response will be extracted from X
    #    if not NULL, effectively running a multivariate lasso regression Y~X
    # {param,matrix} X: regressor matrix/time series observations stacked in rows
    # {param,double} d: VAR(d) model will be estimated if d is not NULL
    # {param,vector} lambda_seq: sequence of tuning parameter for lasso estimate
    # {param,vector} penalty.factor: vector that allows different coordinates to get different penalty
    # {param,boolean} refit: whether do refit after obtaining a low-dim support
    
    # rtype: a list of two elements A (col-concatenated version), Alist
    
    n = nrow(X); p = dim(X); X_copy = X;
    if (is.null(Y))
    {
        if (is.null(lag))
            stop("Error in Est_VAR(): response matrix is not specified. d must be non NULL.\n");
        
        n_initial = nrow(X);
        Y = X[(d+1):n,];
        Xlist = list();
        for (i in 1:d)
            Xlist[[i]] = X[(d-i+1):(n-i),]
        X = do.call("cbind",Xlist);
    }
    else
    {
        if (!is.null(d))
            warning("Both X and Y are provided; d is set to NULL.\n");
        d = NULL;
    }
    
    dimX = ncol(X); dimY = ncol(Y);
    stopifnot( nrow(X) == nrow(Y) );
    
    if (is.null(lambda_seq))
    {
        warning("lambda_seq is null, set to seq(0.1,1,by=0.1)*sqrt(log(p)/n).\n")
        lambda_seq = seq(from=0.1,to=1,by=0.1)*sqrt(log(dimX)/n);
    }
    
    bic_vec = rep(0,length(lambda_seq))
    for (k in 1:length(lambda_seq))
    {
        temp = sparse_VAR_est(Y=Y,X=X,lambda=lambda_seq[k],penalty.factor=penalty.factor,refit=refit);
        bic_vec[k] = BIC_calc(Y,X,temp$A,penalty="lasso",lambda=lambda_seq[k]);
    }
    
    id = which(bic_vec==min(bic_vec))[1];
    lambda_active = lambda_seq[id];
    out = sparse_VAR_est(Y,X,lambda=lambda_active,penalty.factor=penalty.factor,refit=refit);
    
    return(list(BIC_id=id));
}
