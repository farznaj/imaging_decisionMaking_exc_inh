clear;
% same demo as demo_script.m but using the clas @Sources2D
%% load file

% addpath(genpath('../constrained-foopsi')); % FN commented
% addpath(genpath('utilities')); % FN commented
             
doplots = false ; % FN
% Read tiff files. % FN
mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;

%{
convert2double = 1;
Y = tifToMat(mousename, imagingFolder, mdfFileNumber, ch2read, convert2double);
Y = Y{ch2read};
%}

tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);
Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end


%{
nam = 'demoMovie.tif';          % insert path to tiff stack here
sframe=1;						% user input: first frame to read (optional, default 1)
num2read=2000;					% user input: how many frames to read   (optional, default until the end)

Y = bigread2(nam,sframe,num2read);
%}
if ~isa(Y,'double');    Y = double(Y);  end         % convert to double


[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% Set parameters

K = 200; % 30;        % FN                                   % number of components to be found
tau = 4;                                          % std of gaussian kernel (size of neuron) 
p = 2;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;                                  % merging threshold

obj = Sources2D;
updateParams(obj,...            
    'd1',d1,'d2',d2,...                         % dimensions of datasets
    'search_method','ellipse','dist',3,...      % search locations when updating spatial components
    'deconv_method','constrained_foopsi',...    % activity deconvolution method
    'temporal_iter',2,...                       % number of block-coordinate descent steps 
    'fudge_factor',0.98,...                      % bias correction for AR coefficients
    'merge_thr',merge_thr,...                    % merging threshold
    'ssub', 2,... % FN
    'tsub', 3 ... % FN
    );

%% Data pre-processing

Y = preprocess(obj,Y,p);

%% fast initialization of spatial components using greedyROI and HALS

center = initComponents(obj, Y, K, tau);

if doplots
% display centers of found components
Cn =  correlation_image(Y); %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
figure;imagesc(Cn);
    axis equal; axis tight; hold all;
    scatter(center(:,2),center(:,1),'mo');
    title('Center of ROIs found from initialization algorithm');
    drawnow;
end

%% update spatial components
Yr = reshape(Y,d,T);
clear Y;
updateSpatial(obj, Yr);

%% update temporal components

Y_res = updateTemporalParallel(obj, Yr); % FN updateTemporal to parallel.
% nv = norm(obj.C - objp.C, 'fro') / norm(obj.C, 'fro');

%% merge found components
Apr = obj.A;    % store non-merged components
Cpr = obj.C;
[K_m, merged_ROIs] = merge(obj, Y_res);
display_merging = 1; % flag for displaying merging example
if doplots && display_merging
    i = 1; randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
    figure;
        set(gcf,'Position',[300,300,(ln+2)*300,300]);
        for j = 1:ln
            subplot(1,ln+2,j); imagesc(reshape(Apr(:,merged_ROIs{i}(j)),d1,d2)); 
                title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
        end
        subplot(1,ln+2,ln+1); imagesc(reshape(obj.A(:,K_m-length(merged_ROIs)+i),d1,d2));
                title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
        subplot(1,ln+2,ln+2);
            plot(1:T,(diag(max(Cpr(merged_ROIs{i},:),[],2))\Cpr(merged_ROIs{i},:))'); 
            hold all; plot(1:T,obj.C(K_m-length(merged_ROIs)+i,:)/max(obj.C(K_m-length(merged_ROIs)+i,:)),'--k')
            title('Temporal Components','fontsize',16,'fontweight','bold')
        drawnow;
end

%% repeat
updateSpatial(obj, Yr);
Y_res = updateTemporalParallel(obj, Yr); % FN updateTemporal to parallel.
% [C_df, ~, S_df] = extractDF_F(obj, Yr, K_m+1); % FN

%% do some plotting
[srt] = orderROIs(obj);     % order components

[C_df, ~, S_df] = extractDF_F(obj, Yr, K_m+1); % FN moved it here so C_df and S_df are also ordered.

if doplots
    contour_threshold = 0.95;   % amount of energy used for each component to construct contour plot
    figure;
    [json_file] = viewContours(obj, Cn, contour_threshold, 1);
    pause; 
    %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)
    plotComponents(obj, Yr, Cn);     % display all components
end

%% make movie
if doplots
    makePatchVideo(obj, Yr) 
end


%% FN
save('demo_script_parallel_fni17_151102_001', 'obj')

%}