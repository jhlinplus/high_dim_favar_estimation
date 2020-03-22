## ********************************************************
## Description: Regularized VAR(d) least suqare estimation
##
## Regularized VAR estimation with LS
## >> regularied_VAR_est(): vanilla VAR(d) sparse/ridge estimation with specified penalty parameter
## >> regularized_VAR_bic(): VAR(d) estimation based on BIC over a sequence of tuning paarmeters
## >> regularized_VAR_validate(): VAR(d) estimation based on rolling window validation over a sequence of tuning parameters
## >> regularized_VAR_auto(): "automatic" estimation. use with caustion
##
## Author: Jiahe Lin
## ********************************************************
pkgs = c('glmnet','doParallel');
new = pkgs[!(pkgs%in%installed.packages()[,"Package"])]
if (length(new)){
    for (pkg in new)
    {
        install.packages(pkg, dependencies = TRUE)
    }
}
sapply(pkgs, require, character.only = TRUE);
registerDoParallel(cores=detectCores(all.tests = FALSE, logical = TRUE)-1);
cat(sprintf("Total number of workers = %d.\n",getDoParWorkers()));

## load BIC calculation
source("_LIB_IC_Calc.R");

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# Regularized estimate of a VAR with the given penalty parameter lambda
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
regularized_VAR_est = function(Y=NULL,X,d=NULL,lambda,penalty.factor=NULL,alpha=1,refit=FALSE,parallel=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
	#	if NULL, then response will be extracted from X based on the specification of d
	#	if not NULL, effectively running a sparse multivariate lasso regression Y~X
	# {param,matrix} X: regressor matrix/time series observations stacked in rows
	# {param,double} d: VAR(d) model will be estimated if d is not NULL
	# {param,double} lambda: penalty/tuning parameter for lasso estimate
	# {param,vector} penalty.factor: vector that allows different coordinates to get different penalty
    # {param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
	# {param,boolean} refit: whether do refit after obtaining a low-dim support 
	# {param,boolean} parallel: whether run parallel when doing lasso/ridge regressions
    
	# {return, list} the return value has the following two components:
    # {$A,matrix} col-concatenated version of Alist, of size p by (p*d)
    # {$Alist,list} each element corresponding to the coef estimate of the lag
    
	n = nrow(X); p = ncol(X);
    
    is_Y_provided = !is.null(Y);
    if (!is_Y_provided)
	{
    	if (is.null(d))
			stop("Error in regularized_VAR_est(): response matrix is not specified. d must be non NULL.\n");
		
		Y = X[(d+1):n,];
		Xlist = list();
		for (i in 1:d)
			Xlist[[i]] = X[(d-i+1):(n-i),]
		X = do.call("cbind",Xlist);
    }
	else
	{
		if (!is.null(d))
            warning("Both X and Y are provided; d will not be used for extracting regressors/responses, but will be used for column-partitioning the final estimate.\n")
        else
        {
            d = 1;
        }
	}

	dimX = ncol(X); dimY = ncol(Y);
	stopifnot( nrow(X) == nrow(Y) );
	
	if (is.null(penalty.factor))
	{
        A = array(0,c(dimY,dimX));
        if (parallel)
        {
            Arow_list = foreach(j = 1:dimY,.packages=pkgs) %dopar%
            {
                temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE,alpha=alpha);
                Aj = as.numeric(temp$beta);
            }
            for (j in 1:dimY)
            {
                A[j,] = Arow_list[[j]];
            }
        }
        else
        {
            for (j in 1:dimY)
            {
                temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE,alpha=alpha);
                A[j,] = as.numeric(temp$beta);
            }
        }
    }
    else ## allow for different penalty across coordinates
	{
        stopifnot(length(penalty.factor)==dimX);
		A = array(0,c(dimY,dimX));
        
        if (parallel)
        {
            Arow_list = foreach(j = 1:dimY,.packages=pkgs) %dopar%
            {
                temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE,penalty.factor=penalty.factor,alpha=alpha);
                Aj = as.numeric(temp$beta);
            }
            for (j in 1:dimY)
            {
                A[j,] = Arow_list[[j]];
            }
        }
        else
        {
            for (j in 1:dimY)
            {
                temp = glmnet(x=X,y=Y[,j],lambda=lambda,intercept=FALSE,penalty.factor=penalty.factor,alpha=alpha);
                A[j,] = as.numeric(temp$beta);
            }
        }
        
    }
    
    if (refit)
	{
        skeleton = 1*(A!=0);
        for (j in 1:dimY)
		{
            skeleton_temp = which(skeleton[j,]!=0);
            if ( length(skeleton_temp) < nrow(X) && length(skeleton_temp) > 0 )
            {
                A[j,skeleton_temp] = as.numeric(lm(Y[,j]~X[,skeleton_temp]+0)$coef);
            }
            else
                next;
        }
    }

	col_size = round(ncol(A)/d);
	stopifnot(col_size*d==ncol(A));
	
    Alist = lapply(1:d,function(xx){A[,(1+(xx-1)*col_size):(xx*col_size)]});
    names(Alist) = paste("d=",1:d,sep="");
	
    return(list(A=A,Alist=Alist));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
# Regularized VAR with auto-selected tuning parameters using bic
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
regularized_VAR_bic = function(Y=NULL,X,d=NULL,lambda_seq=NULL,penalty.factor=NULL,alpha=1,refit=FALSE,parallel=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
    #    if NULL, then response will be extracted from X
    #    if not NULL, effectively running a multivariate lasso regression Y~X
    # {param,matrix} X: regressor matrix/time series observations stacked in rows
    # {param,double} d: VAR(d) model will be estimated if d is not NULL
    # {param,vector} lambda_seq: the sequence of tuning parameters for lasso estimate
    # {param,vector} penalty.factor: vector that allows different coordinates to get different penalty
    # {param,double} alpha: alpha in glmnet. alpha = 1: lasso; alpha = 0: ridge
    # {param,boolean} refit: whether do refit after obtaining a low-dim support
    # {param,boolean} parallel: whether run parallel when doing lasso/ridge regressions
    
    # {return, list} the return value has the following two components:
    # {$idx,double} the idx of the BIC selected
    # {$output,list} the output from the final sparse VAR estimation. see documentation for sparse_VAR_est()
    
    n = nrow(X); 
    
    is_Y_provided = !is.null(Y);
    if (!is_Y_provided)
    {
        if (is.null(d))
            stop("Error in regularized_VAR_bic(): response matrix is not specified. d must be non NULL.\n");
        
        Y = X[(d+1):n,];
        Xlist = list();
        for (i in 1:d)
            Xlist[[i]] = X[(d-i+1):(n-i),]
        X = do.call("cbind",Xlist);
    }
    else
    {
        if (!is.null(d))
            warning("Both X and Y are provided; d will not be used for extracting regressors/responses, but will be used for column-partitioning the final estimate.\n")
        else
        {
            d = 1
        }
    }
    
    dimX = ncol(X); 
    stopifnot( nrow(X) == nrow(Y) );
    
    if (is.null(lambda_seq))
    {
        warning("lambda_seq is null, set to seq(0.1,1,by=0.1)*sqrt(log(p)/n).\n")
        lambda_seq = seq(from=0.1,to=1,by=0.1)*sqrt(log(dimX)/n);
    }
    
    bic_vec = rep(0,length(lambda_seq))
    for (k in 1:length(lambda_seq))
    {
        ## note that during selection step, refitting is disabled
        temp = regularized_VAR_est(Y=Y,X=X,d=NULL,lambda=lambda_seq[k],penalty.factor=penalty.factor,alpha=alpha,refit=FALSE,parallel=parallel);
        
        bic_vec[k] = BIC_calc(Y,
                              X,
                              temp$A,
                              penalty=ifelse(alpha==1,'lasso',ifelse(alpha==0,'ridge',NULL)),
                              lambda=lambda_seq[k]);
    }
    
    id = which(bic_vec==min(bic_vec))[1];
    lambda_active = lambda_seq[id];
    
    out = regularized_VAR_est(Y=Y,X=X,d=d,lambda=lambda_active,penalty.factor=penalty.factor,alpha=alpha,refit=refit,parallel=parallel);
    
    ## return the idx of the BIC selected and the output from the final sparse VAR estimation
    return(list(idx=id,out=out));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Regularized estimate of a VAR with auto-selected tuning parameter
# based on 1-step-ahead rolling window validation
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
regularized_VAR_validate = function(Y=NULL,X,d=NULL,lambda_seq=NULL,penalty.factor=NULL,alpha=alpha,
        train_size=0.8,validate_size=0.1,refit=FALSE,parallel=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
    #    if NULL, then response will be extracted from X
    #    if not NULL, effectively running a multivariate lasso regression Y~X
    # {param,matrix} X: regressor matrix/time series observations stacked in rows
    # {param,double} d: VAR(d) model will be estimated if d is not NULL
    # {param,vector} lambda_seq: the sequence of tuning parameters for lasso estimate
    # {param,vector} penalty.factor: vector that allows different coordinates to get different penalty
    # {param,double} train_size/validate_size: the size of training and validation set size
    #   Note that the number of windows will be selected as 1 - train_size - validate_size
    # {param,boolean} refit: whether do refit after obtaining a low-dim support
    # {param,boolean} parallel: whether run parallel when doing lasso/ridge regressions
    
    # {return, list} the return value has the following two components:
    # {$idx,double} the idx of the tunning parameter selected
    # {$output,list} the output from the final sparse VAR estimation. see documentation for sparse_VAR_est()

    n = nrow(X); 
    
    is_Y_provided = !is.null(Y);
    if (!is_Y_provided)
    {
        if (is.null(d))
            stop("Error in regularized_VAR_validate(): response matrix is not specified. d must be non NULL.\n");
        
        Y = X[(d+1):n,];
        Xlist = list();
        for (i in 1:d)
            Xlist[[i]] = X[(d-i+1):(n-i),]
        X = do.call("cbind",Xlist);
    }
    else
    {
        if (!is.null(d))
            warning("Both X and Y are provided; d will not be used for extracting regressors/responses, but will be used for column-partitioning the final estimate.\n")
        else
        {
            d = 1
        }
    }
    
    dimX = ncol(X); effective_n = nrow(X);
    stopifnot( nrow(X) == nrow(Y) );
    
    if (is.null(lambda_seq))
    {
        warning("lambda_seq is null, set to seq(0.1,1,by=0.1)*sqrt(log(p)/n).\n")
        lambda_seq = seq(from=0.1,to=1,by=0.1)*sqrt(log(dimX)/n);
    }
    
    train_size = round(effective_n*train_size);
    validate_size = round(effective_n*validate_size);
    window_number = effective_n - train_size - validate_size;
    cat(sprintf("train size = %d, test size = %d, validation window number = %d.\n", train_size, validate_size, window_number))
    
    validate_err = array(0,c(length(lambda_seq),window_number));
    
    for (k in 1:length(lambda_seq))
    {
        lambda_temp = lambda_seq[k];
        
        for (i in 1:window_number)
        {
            # estimation
            Y_est = Y[i:(i+train_size-1),]; 
            X_est = X[i:(i+train_size-1),];
            temp_est = regularized_VAR_est(Y=Y_est,X=X_est,d=NULL,lambda=lambda_temp,penalty.factor=penalty.factor,alpha=alpha,refit=FALSE,parallel=parallel);
            
            # validation
            Y_vld = Y[(i+train_size):(i+train_size+validate_size-1),];
            X_vld = X[(i+train_size):(i+train_size+validate_size-1),];
            # validation error
            ErrMat = Y_vld - X_vld %*% t(temp_est$A);
            validate_err[k,i] = mean( apply(ErrMat^2,1,sum)/apply(Y_vld^2,1,sum) )
        }
    }
    validate_err = apply(validate_err,1,mean);
    
    id = which.min(validate_err)
    lambda_active = lambda_seq[id];
    
    output = regularized_VAR_est(Y,X,d=d,lambda=lambda_active,penalty.factor=penalty.factor,alpha=alpha,refit=refit,parallel=parallel);
    ## return the idx of the BIC selected and the output from the final sparse VAR estimation
    return(list(idx=id,out=output));
}

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Auto regularized estimation of a VAR -- use with caution
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
regularized_VAR_auto = function(Y=NULL,X,d=NULL,alpha=1,lambda_seq=NULL,selection="bic",parallel=FALSE,refit=FALSE)
{
    # {param,matrix} Y: response vectors stacked in rows
    #    if NULL, then response will be extracted from X
    #    if not NULL, effectively running a multivariate lasso regression Y~X
    # {param,matrix} X: regressor matrix/time series observations stacked in rows
    # {param,double} d: VAR(d) model will be estimated if d is not NULL
    # {param,vector} lambda_seq: the sequence of tuning parameters for lasso estimate. can be NULL
    # {param,string} selection: "bic" or "validation"
    # {param,boolean} parallel: whether run parallel when doing lasso/ridge regressions
    
    if (selection == "bic")
    {
        out = regularized_VAR_bic(Y=Y,X=X,d=d,lambda_seq=lambda_seq,alpha=alpha,refit=refit,parallel=parallel);
    }
    else if (selection == "validation")
    {
        out = regularized_VAR_validate(Y=Y,X=X,d=d,lambda_seq=lambda_seq,alpha=alpha,train_size=0.8,validate_size=0.1,refit=refit,parallel=parallel);
    }
    return(out)
}
