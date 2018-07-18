function [traces_wheel, times_wheel] = wheelInfo(alldata)
% [traces_wheel, times_wheel] = wheelInfo(alldata)
% Run script wheelAnalysis to get some plots and better understand the wheel data.
% remember in the arduino code the max length for the rotary position array
% is defined 2500, which at 100Hz will be 25sec.... so if you see a
% mismatch in the length of trial (computed from bcontrol or imaging data)
% and times_wheel it could be due to this upper limit of 25 sec!



for itr = 1:length(alldata)
    wheelTimeRes = alldata(itr).wheelSampleInt;
    alldata(itr).wheelTimes = wheelTimeRes/2 +  wheelTimeRes * (0:length(alldata(itr).wheelRev)-1);
end
times_wheel = {alldata.wheelTimes};


traces_wheel = {alldata.wheelRev};


% since the absolute values don't matter, I assume the position of the
% wheel at the start of a trial was 0. I think values are negative, bc when
% the mouse moves forward, rotary turns counter clock wise.

% traces_wheel = cellfun(@(x)x-x(1), traces_wheel, 'uniformoutput', 0);



%% Take care of the reset in rotary values.

% a = cell2mat({alldata.wheelRev}');

traces_wheel_new = traces_wheel;

for ii = 1:length(alldata)
    a = alldata(ii).wheelRev;
    
    % xlabel('Rotary samples')
    % ylabel('Rotary position')
    
    % Rotary position seems to reset (to +16) once it reaches -16. We need to
    % take care of this! You do this by subtracting out the amount of jump from
    % point ~-16 to ~16:
%     f = find(round(diff(a)) > 16);
    f = find(round(abs(diff(a))) > 16); % using abs, bc sometimes it goees to -16 from 16 soon after a reset!
    
    if ~isempty(f)    
%         figure('name', num2str(ii)); hold on
%         plot(a)
        fprintf('Rotary seems to be reset during trial %i\n', ii)
        for i = 1:length(f)-1
            r = f(i)+1 : f(i+1);
            d = a(f(i)+1) - a(f(i));
            a(r) = a(r) - d;
        end
        
        i = length(f);
        d = a(f(i)+1) - a(f(i));
        a(f(i)+1 : end) = a(f(i)+1 : end) - d;
        
%         plot(a)
%         legend('raw','after controling for the reset')
        
        traces_wheel_new{ii} = a;        
    end    
end


%%
traces_wheel = traces_wheel_new;

% since the absolute values don't matter, I assume the position of the
% wheel at the start of the session was 0. I think values are negative, bc
% when the mouse moves forward, rotary turns counter clock wise.
traces_wheel = cellfun(@(x)x-x(1), traces_wheel, 'uniformoutput', 0);


