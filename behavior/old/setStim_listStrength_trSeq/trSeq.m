% function from of script trial_sequence_thconsec

% method1 for removing more than th_consecutive trials of the same type in a row
% remember this method works fine however, you still end up with some small
% percentage of trials above th_consecutive. This is because although you
% remove parts of the coin sequence that have >th_consecutive in a row, but
% after removal, some new parts are created that have this problem.
% for this reason choose th_consecutive 1 below what you actually wish.

%{
max_trials = 1E6;
th_consecutive = 4;  % maximum number of consecutive trials of the same type that you want to have.
%%% Original case: any number of trials of the same type can happen in a row
coin = round(rand(1,max_trials));
%}

%%
function coin_th = trSeq(coin, th_consecutive)
% identify consecutive 0s > th_consectuve
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
coin_th = coin;




