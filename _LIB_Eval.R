#######################################################################
## Description: Evaluate the performance of the sparse and 
##      the low rank component coming out of the estimation
##
## This is used when synthetic data is generated and the truth is known
#######################################################################

Eval_Sparse = function(true,est)
{
    
    true = as.matrix(true);
    est = as.matrix(est);
    
    est.v = as.vector(!!est);
    true.v = as.vector(!!true);
    
    TP = sum(est.v & true.v);
    TN = sum((!est.v) & (!true.v));
    FP = sum(est.v & !true.v);
    FN = sum(!est.v & true.v);

    ## evaluate SEN, SPC and Error in Fnorm for a matrix
    SEN = TP/(TP+FN); # aka true positive rates
    SPC = TN/(TN+FP); # aka true negative rate;
    Error = norm(true-est,"F")/norm(true,'F')
    
    result = c(SEN,SPC,Error); 
    names(result) = c("SEN","SPC","ERR");    
    
    return(result)   
}

Eval_Flats = function(truth,est)
{
    
    truth = as.matrix(truth);
    est = as.matrix(est);
    
    # Evaluate the performance of the estimated hyperplane, through the following metrics:
    #   >> the estimated rank
    #   >> principal angles and sin-theta distance (http://www.stat.cmu.edu/~arinaldo/Teaching/36755/F17/Scribed_Lectures/F17_1023.pdf")
    #   >> frobenius norm error
    
    QR1 = qr(truth);
    QR2 = qr(est);
    SVD = svd(t(qr.Q(QR2)) %*% qr.Q(QR1));
    
    options(warn=-1)
    acos_val = acos(SVD$d); ## NaN value will be produced in the case where the singular value exceeds 1 due to numerical reasons
    options(warn=0)
    
    for (i in 1:length(acos_val))
    {
        if (is.nan(acos_val[i]))
        {
            acos_val[i] = 0;
        }
    }
    sin_thetaD_sq = sum( (sin(acos_val))^2); ## 1/2* || Proj_true - Proj_est ||_F^2
    fnorm_proj_true = norm( qr.Q(QR1) %*% t(qr.Q(QR1)), 'f' );
    # relative projection error
    relative_proj_error = sqrt(2*sin_thetaD_sq)/fnorm_proj_true;
    # relative frobenius norm error
    relative_ferror = norm(truth-est,'f')/norm(truth,'f');
    
    metrics = c(rankMatrix(est),sqrt(sin_thetaD_sq),relative_proj_error,relative_ferror);
    names(metrics) = c("estimated_rank","sin_theta_distance","rel_proj_err","rel_fnorm_err");
    
    return( list(prcp_angles = acos_val, metrics = metrics) )
}
