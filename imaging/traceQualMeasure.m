function [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(C_df, S_df)
% [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(C_df, S_df)
%
% Compute two measures of signal/noise (avePks2sdS, aveProm2sdS) to evaluate trace quality


%%
avePks2sdS = NaN(1, size(S_df,1));
aveProm2sdS = NaN(1, size(S_df,1));
% avePks = NaN(1, size(S_df,1));
% aveProm = NaN(1, size(S_df,1));

for rr = 1:size(S_df,1)
    bl = quantile(C_df(rr,:), .1);       
    [pks,~, ~, p] = findpeaks(C_df(rr,:), 'minpeakheight', bl+.05); %'minpeakdistance', 3, 'threshold', .005);

    sn = S_df(rr,:);
    sn = sn/max(sn);

%     avePks(rr) = mean(pks);
%     aveProm(rr) = mean(p);
    
    avePks2sdS(rr) = mean(pks)/std(sn);
    aveProm2sdS(rr) = mean(p)/std(sn);             
end



% both mean(pks)std(sn) and mean(pks)*mean(p)/std(sn) seem to be good and a threshold of <3 would catch most bad neurons.
% I think this is the best for defining bad quality neurons: (mean(pks)/std(sn)<3 |  mean(p)/std(sn) < 1 )


measQual = (avePks2sdS-3) .* (aveProm2sdS-1);
s = sign(avePks2sdS-3)<0 & sign(aveProm2sdS-1)<0;
measQual(s) = -measQual(s);





