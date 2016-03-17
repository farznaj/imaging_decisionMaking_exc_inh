function [u, l] = wilsonBinomialConfidenceInterval(p, z, n)
% [upper, lower] = wilsonBinomialConfidenceInterval(p, z, n)
%
% Implementation of the Wilson Binomial Proportion Confidence Interval.
% p is the proportion of successes.
% z is the z_(1-alpha/2), i.e., the 1 alpha / 2 percentile of a standard
%   normal distribution, with alpha the error percentile. For example, for 
%   a 95% confidence level the error (alpha) is 5%, so 
%   1 alpha / 2 = 0.975 and z_(1 alpha/2) = 1.96.
% n is the sample size.
%
% upper and lower are the confidence bounds.

u = (p + z^2/(2*n) + z * sqrt(p*(1-p)/n + z^2/(4*n^2))) / (1 + z^2/n);
l = (p + z^2/(2*n) - z * sqrt(p*(1-p)/n + z^2/(4*n^2))) / (1 + z^2/n);
