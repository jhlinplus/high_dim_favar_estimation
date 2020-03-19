## ***************************************************
## Description: library for calculating information criteria
##
## Author: Jiahe Lin
## ***************************************************

## PIC calculation a la Ando and Bai (2015 Econometric Review)
PIC_calc = function(Y,X,B,Theta,r)
{
    n = nrow(Y); q = ncol(Y); 
    
    RSS = norm(Y - X %*% t(B) - Theta,'f')^2;
    sigma_hat = RSS/(n*q)
    
    df_B = sum(!!B);
    penalty = sigma_hat* ( log(n)/(n*q)*df_B + r*(n+q)/(n*q)*log(n*q) ); 
    
    PIC = sigma_hat + penalty; 
    
    return(list(PIC=PIC,sigma_hat=sigma_hat,penalty=penalty));
}

## BIC calculation for multivariate regression
BIC_calc = function(Y,X,B,penalty="ridge",lambda=NULL)
{
    n = nrow(Y);
    RSS = apply( Y - X %*% t(B) , 2 , function(x){sum(x^2)});
    
    if (penalty=="lasso")
    {
        return(sum(log(RSS)) + log(n)/n*sum(!!B));
    }
    else if (penalty=="ridge")
    {
        sigval = svd(X)$d; lambda0 = n*lambda;
        df = ncol(Y)* sum( sigval^2/(sigval^2 + rep(lambda0,length(sigval))) );
        return(sum(log(RSS)) + log(n)/n*df);
    }
    else
    {
        stop("Error in BIC_Calc(): unrecognized penalty")
    }
}

## Information Criterion 1 for selecting the numbers of factors (IC1 in Bai, Econometria 2002)
Bai_IC = function(Y,r)
{
    n = nrow(Y); q = ncol(Y);
    
    Theta_SVD = svd(Y);
    Theta = Theta_SVD$u %*% diag(c(Theta_SVD$d[1:r],rep(0,length(Theta_SVD$d)-r))) %*% t(Theta_SVD$v);
    
    sigma_hat_sq = apply((Y-Theta)^2,2,sum)/n;
    V = sum(sigma_hat_sq)/q;
    
    IC1_Bai = log(V) + r*(q+n)/(q*n)*log(n*q/(n+q));
    return(IC1_Bai)
}
