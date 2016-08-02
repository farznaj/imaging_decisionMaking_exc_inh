function  Yp = predictLogisticRegression(x, w, b, lps)
    XW = x*w(:);
    prob1Expression = lps + (1 - 2 * lps)./(1.0 + exp(- (XW + b))); % expression of logistic function (prob of 1)
    Yp = prob1Expression>0.5;
end