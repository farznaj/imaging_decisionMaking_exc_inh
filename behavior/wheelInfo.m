function [traces_wheel, times_wheel] = wheelInfo(alldata)
% [traces_wheel, times_wheel] = wheelInfo(alldata)

for itr = 1:length(alldata)
    wheelTimeRes = alldata(itr).wheelSampleInt;
    alldata(itr).wheelTimes = wheelTimeRes/2 +  wheelTimeRes * (0:length(alldata(itr).wheelRev)-1);
end

% since the absolute values don't matter, I assume the position of the wheel at the start of a trial was 0. I think values are negative, bc when the mouse moves forward, rotary turns counter clock wise.
traces_wheel = {alldata.wheelRev};
traces_wheel = cellfun(@(x)x-x(1), traces_wheel, 'uniformoutput', 0); 

times_wheel = {alldata.wheelTimes};