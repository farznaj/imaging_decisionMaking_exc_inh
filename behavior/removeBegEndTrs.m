function alldata = removeBegEndTrs(all_data, thbeg)
% thbeg : exclude thbeg initial trials of each session.

alldata = all_data(1:end-1);        

if length(alldata)>thbeg
    alldata(1:thbeg) = [];
end

