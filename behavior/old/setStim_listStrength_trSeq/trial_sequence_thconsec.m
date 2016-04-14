% make sure there are no more than th_consectuive trials of the same type
% in a row

% concern: inducing switch bias.

% if th_consecutive is 3, there will be 60% switch, if it is 4, 56% switch, random case: 50% switch
% (ie left followed by right and vice versa)

% If you want to be very conservative: I would choose th_consecutive 1 less than what you wish to have. ie. if
% you want not to have >3 consecutive trials, I would choose 2 for
% th_consecutive. This is bc of what explained below.
max_trials = 1E6;
th_consecutive = 4;  % maximum number of consecutive trials of the same type that you want to have.


%%% Original case:
% any number of trials of the same type can happen in a row
coin = round(rand(1,max_trials));
ccoin = coin; % save a backup

%% look at the distribution of number of trials of the same type in a row.
c = [1 coin 1];
f1 = find(c==1);
n0s_cons = diff(f1)-1;

c = [0 coin 0];
f1 = find(c==0);
n1s_cons = diff(f1)-1;
[max(n0s_cons)  max(n1s_cons)]

mm = max([max(n0s_cons), max(n1s_cons)]);
mn = min([min(n0s_cons), min(n1s_cons)]);
v = [mn:1:mm];
v = v+.5; % use this vector if you don't want to look at percentage of 0
% events in a row.
n0 = histc(n0s_cons, v); 
n1 = histc(n1s_cons, v); 
figure; 
% plot(v, [n0/sum(n0) ; n1/sum(n1)]')
plot(v+.5, [n0/sum(n0) ; n1/sum(n1)]')

legend(sprintf('%s%0.2f', 'fraction of switches: ', nanmean(abs(diff(coin))==1)))

%% method1 for removing more than th_consecutive trials of the same type in a row
% remember this method works fine however, you still end up with some small
% percentage of trials above th_consecutive. This is because although you
% remove parts of the coin sequence that have >th_consecutive in a row, but
% after removal, some new parts are created that have this problem.
% for this reason choose th_consecutive 1 below what you actually wish.

tic
% identify consecutive 0s > th_consectuve
% evorno = coin;
evd = coin;
%         evd(end+1) = 1;
evd = [1 evd 1];
f1 = find(evd==1)-1; % you subtract 1 bc you added an element to the beginning of evd.
evdist = diff(f1)-1;

f11 = f1(find(evdist > th_consecutive )); 
f12 = f1(find(evdist > th_consecutive)+1);
f_ldist = [f11;f12]';


% identify consecutive 1s > th_consecutive
evd = coin;
evd(coin==0) = 1;
evd(coin==1) = 0;

evd = [1 evd 1];
f1 = find(evd==1)-1;
evdist = diff(f1)-1;

f11 = f1(find(evdist > th_consecutive )); 
f12 = f1(find(evdist > th_consecutive)+1);
f_ldist1 = [f11;f12]';


% remove consecutive 0s and 1s
for fi = 1:size(f_ldist,1)
    coin(f_ldist(fi,1)+1 : f_ldist(fi,2)) = NaN;        
end

for fi = 1:size(f_ldist1,1)
    coin(f_ldist1(fi,1)+1 : f_ldist1(fi,2)) = NaN;        
end

coin(isnan(coin)) = [];

length(coin)/length(ccoin)

t1 = toc



%% method0 (wrong)
% This method is not correct (and also slow!). because it doesn't give a decreasing
% exponential of number of [trials of the same type in a row], which is
% normally expected to happen. This is because you go up to th_consecutive
% [trials of the same type in a row], and after that you don't allow one
% more of the same type to happen. So you get a false increase in the
% number of th_consecutive [trials of the same type in a row]. 

tic

coin = NaN(1, max_trials);
i = 2;
while i<max_trials
    coin(i) = round(rand);
    if (coin(i)-coin(i-1))~=0 % switch in trial type has occured.
        n0 = 0;
        n1 = 0;
    end
    
    if coin(i)==0
        n0 = n0+1;
    else
        n1 = n1+1;
    end
    
    if n0 == th_consecutive % switch needed on the next trial bc there has been more than th_consectuve trials of the same type in a row.
        coin(i+1) = 1;
        n0 = 0;
        n1 = 1;
        i = i+2;
    elseif n1 == th_consecutive
        coin(i+1) = 0;
        n0 = 1;
        n1 = 0;
        i = i+2;
    else
        i = i+1;
    end
end

coin(isnan(coin)) = [];
        

t0 = toc


