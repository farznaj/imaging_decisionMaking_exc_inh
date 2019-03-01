%{
normX = 1;

frameLength = 1000/30.9; % sec.
regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.

%%
eventI_initTone0 = eventI_initTone;
traces_aligned_fut_initTone0 = traces_aligned_fut_initTone;
[eventI_initTone, traces_aligned_fut_initTone] = downsamp_x(eventI_initTone0, traces_aligned_fut_initTone0, regressBins, normX);

eventI_stimOn0 = eventI_stimOn; 
traces_aligned_fut_stimOn0 = traces_aligned_fut_stimOn;
[eventI_stimOn, traces_aligned_fut_stimOn] = downsamp_x(eventI_stimOn0, traces_aligned_fut_stimOn0, regressBins, normX);

eventI_1stSideTry0 = eventI_1stSideTry;
traces_aligned_fut_1stSideTry0 = traces_aligned_fut_1stSideTry;
[eventI_1stSideTry, traces_aligned_fut_1stSideTry] = downsamp_x(eventI_1stSideTry0, traces_aligned_fut_1stSideTry0, regressBins, normX);

eventI_reward0 = eventI_reward;
traces_aligned_fut_reward0 = traces_aligned_fut_reward;
[eventI_reward, traces_aligned_fut_reward] = downsamp_x(eventI_reward0, traces_aligned_fut_reward0, regressBins, normX);

%}

function [eventI_ds, X_svm, time_trace_d] = downsamp_x(eventI, X_svmo, regressBins, normX)
%%
%%%%%%%%%% set frames before frame0 (not including it)
e = eventI-1;
f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
x = X_svmo(f,:,:); % X_svmo including frames before frame0
[T1, N1, C1] = size(x);
tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
xdb = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames before frame0
a = nan(1,size(xdb,1),size(xdb,2));
if tt==1
    a(1,:,:) =  xdb;
    xdb = a;
end

%%%%%%%%%%% set frames after frame0 (including it)
lenPost = size(X_svmo,1) - (eventI-1);
f = eventI : (eventI-1) + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins
%             f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins
x = X_svmo(f,:,:); % X_svmo including frames after frame0
[T1, N1, C1] = size(x);
tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
xda = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames after frame0
a = nan(1,size(xda,1),size(xda,2));
if tt==1
    a(1,:,:) =  xda;
    xda = a;
end

%%%%%%%%%%% set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
X_svm_d = cat(1, xdb, xda);
%             X_svm_d = cat(1, xdb, X_svmo(eventI,:,:), xda);
X_svm = X_svm_d;
fprintf('%d x %d x %d : size(original X_svm)\n', size(X_svmo))
fprintf('%d x %d x %d : size(downsampled X_svm)\n', size(X_svm))


%%%%%%%%%% set downsampled eventI
if size(xda,1)>0
    eventI_ds = size(xdb,1)+1;
else
    eventI_ds = nan;
    warning('After downsampling, there are not enough post frames (so eventI_ds is set to nan!)')
end


%% After downsampling normalize X_svm so each neuron's max is at 1 (you do this in matlab for S traces before downsampling... so it makes sense to again normalize the traces After downsampling so max peak is at 1)

if normX
    % find the max of each neurons across all trials and frames # max(X_svm.flatten())
    m = max(X_svm, [], 3); % max across trials
    m = max(m, [], 1);  % max across frames
    X_svm = bsxfun(@rdivide, X_svm, m);
end



%% Set downsampled time_aligned

frameLength = 1000/30.9; % sec.

totLen = size(X_svm_d,1); %nPreMin + nPostMin +1;
% eventI = min(eventI_allDays);

time_trace = frameLength * ((1 : ceil(regressBins*(totLen+1))) - eventI); % time_trace = time_aligned_1stSide

% set frames before frame0 (not including it)
e = eventI-1;
f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
x = time_trace(f); % time_trace including frames before frame0
T1 = length(x);
tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
xdb = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames before frame0


% set frames after frame0 (including it)
lenPost = length(time_trace) - (eventI-1);
%     lenPost = length(time_trace) - eventI;
f = eventI : (eventI-1) + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
%     f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
x = time_trace(f); % time_trace including frames after frame0
T1 = length(x);
tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
xda = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames after frame0

% set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
time_trace_d = cat(2, xdb, xda);
%     time_trace_d = cat(2, xdb, 0, xda);    
time_trace_d = time_trace_d(1:totLen);
%{
a = -frameLength * (0:nPreMin); a = a(end:-1:1);
b = frameLength * (1:nPostMin);
time_aligned = [a,b];
%}

fprintf('%d : length(original time_trace)\n', length(time_trace))
fprintf('%d : length(downsampled time_trace)\n', length(time_trace_d))







