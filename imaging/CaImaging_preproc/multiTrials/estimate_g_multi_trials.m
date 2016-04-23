function [g,snw] = estimate_g_multi_trials(Y,p,lags,fudge_factor,noise_method)

if nargin < 5 || isempty(noise_method)
    noise_method = 'mean';
end
options.noise_method = noise_method;

if nargin < 4 || isempty(fudge_factor)
    fudge_factor = 1; 0.98;
end

if nargin < 3 || isempty(lags)
    lags = 5;
end

if nargin < 2 || isempty(p)
    p = 2;
end

lags = lags + p;

Nt = length(Y);
XC = zeros(Nt,2*lags+1);
for j = 1:Nt
    XC(j,:) = xcov(Y{j},lags,'biased');
end

Cr =  XC(:,lags:-1:1);
clear XC;
lags = lags - p;
A = zeros(Nt*lags,p);

for i = 1:Nt
        A((i-1)*lags + (1:lags),:) = toeplitz(Cr(i,p:p+lags-1),Cr(i,p:-1:1));
end
gv = Cr(:,p+1:end)';
%g = pinv(A)*gv(:);
g = A\gv(:);

rg = roots([1;-g(:)]);
if ~isreal(rg); rg = real(rg) + .001*randn(size(rg)); end
rg(rg>1) = 0.95 + 0.001*randn(size(rg(rg>1)));
rg(rg<0) = 0.15 + 0.001*randn(size(rg(rg<0)));
pg = poly(fudge_factor*rg);
g = -pg(2:end);

sn = zeros(Nt,1);
w = zeros(Nt,1);
for i = 1:Nt
    sn(i) = get_noise_fft(Y{i}(:)',options);
    w(i) = length(Y{i});
end
snw = sn'*w/sum(w);