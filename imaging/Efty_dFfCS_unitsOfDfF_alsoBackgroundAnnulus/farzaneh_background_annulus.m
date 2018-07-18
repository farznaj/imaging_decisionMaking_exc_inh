% demo script for splitting the field of view in patches and processing in parallel
% with or without memory mapping. See also run_pipeline.m for the complete  
% pre-processing pipeline of large datasets

clear;
gcp;  % start local cluster

%% read file
                                                     
filename = '/Users/epnevmatikakis/Documents/Ca_datasets/Najafi/background/151015_001_001_ch2_MCM.TIF';      
[d1,d2] = size(read_file(filename,1,1));
%% motion correction

motion_correct = true;         % perform motion correction

options_mc = NoRMCorreSetParms('d1',d1,'d2',d2,...
                'init_batch',200,'output_type','tif',...
                'bin_width',200,'max_shift',24,'us_fac',50);

[M,shifts,template,options_mc,col_shift] = normcorre_batch(filename,options_mc);
         

%% memory mapping

is_memmaped = true;
filename = '/Users/epnevmatikakis/Documents/Ca_datasets/Najafi/background/motion_corrected.tif';
if is_memmaped
    if exist([filename(1:end-3),'mat'],'file')
        data = matfile([filename(1:end-3),'mat'],'Writable',true);
    else
        sframe=1;						% user input: first frame to read (optional, default 1)
        num2read=[];					% user input: how many frames to read   (optional, default until the end)
        chunksize=5000;                 % user input: read and map input in chunks (optional, default read all at once)
        data = memmap_file(filename,sframe,num2read,chunksize);
        %data = memmap_file_sequence(foldername);
    end
    sizY = size(data,'Y');                    % size of data matrix
else
    T = 2000;                                 % load only a part of the file due to memory reasons
    data = read_file(filename,1,T);
    sizY = size(data);
end
    

%% Set parameters for CNMF in patches

patch_size = [32,32];                   % size of each patch along each dimension (optional, default: [32,32])
overlap = [6,6];                        % amount of overlap in each dimension (optional, default: [4,4])

patches = construct_patches([d1,d2],patch_size,overlap);
K = 8;                  % number of components to be found
tau = 6;                 % std of gaussian kernel (size of neuron) 
p = 2;                   % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;         % merging threshold

options = CNMFSetParms(...
    'd1',d1,'d2',d2,...
    'nb',1,...                                  % number of background components per patch
    'gnb',2,...                                 % number of global background components
    'ssub',2,...
    'tsub',1,...
    'p',p,...                                   % order of AR dynamics
    'merge_thr',merge_thr,...                   % merging threshold
    'gSig',tau,... 
    'spatial_method','regularized',...
    'cnn_thr',0.2,...
    'patch_space_thresh',0.25,...
    'min_SNR',2);

%% Run on patches

[A,b,C,f,S,P,RESULTS,YrA] = run_CNMF_patches(data,K,patches,tau,p,options);

%% classify components 

rval_space = classify_comp_corr(data,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the space correlation test

try  % matlab 2017b or later is needed for the CNN classifier
    [ind_cnn,value] = cnn_classifier(A,[options.d1,options.d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);
end

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std); % event exceptionality
ind_exc = (fitness < options.min_fitness);

keep = (ind_corr | ind_cnn) & ind_exc;

%% run GUI for modifying component selection (optional, close twice to save values)


%% re-estimate temporal components
A_throw = A(:,~keep);
C_throw = C(~keep,:);
A_keep = A(:,keep);
C_keep = C(keep,:);
options.p = 0;      % perform deconvolution
P.p = 0;
[A2,b2,C2] = update_spatial_components(data,C_keep,f,[A_keep,b],P,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components_fast(data,A2,b2,C2,f,P,options);

%% plot results
figure;
Cn = correlation_image_max(data);  % background image for plotting
plot_contours(A2,Cn,options,1);
plot_components_GUI(data,A2,C2,b,f2,Cn,options);

%% reshape data in 2d
Yr = single(data.Yr);
T = size(Yr,2);

%% compare traces

Yr = single(Yr);
dist = 3;   % distance of annulus from neuron boundary
width = 7;  % width of annulus

ind = randi(size(A2,2));
a_ind = full(reshape(A2(:,ind),d1,d2))>0;                   % find the mask of the component
annulus = imdilate(a_ind,strel('disk',dist));                  
annulus = imdilate(annulus,strel('disk',width)) - annulus;     % create an annulus around it excluding the border.

f_annulus = (annulus(:)'*Yr - (annulus(:)'*A2)*(C2+YrA2))/sum(annulus(:));    % find background contamination using the annulus
c_annulus = a_ind(:)'*Yr/sum(a_ind(:)) - f_annulus; % extract a trace

c_cnmf = C2(ind,:) + YrA2(ind,:);

% calculate a linear fit
c_cnmf = c_cnmf/max(c_cnmf);
c_annulus = c_annulus/max(c_annulus);

l_fit = [sum(c_cnmf.^2), sum(c_cnmf); sum(c_cnmf), T]\[sum(c_cnmf.*c_annulus); sum(c_annulus)];

figure;
    set(gcf,'Position',[200,200,3*480,480])
    subplot(1,3,1);imagesc(reshape(A2(:,ind),d1,d2)/max(A2(:,ind)) + annulus);
        axis square; title(['Component ',num2str(ind)])
    subplot(1,3,2:3);plot(1:T,c_annulus,1:T,l_fit(1)*c_cnmf + l_fit(2))
        legend('annulus','CNMF');
        title(sprintf('Correlation: %1.3f',corr(c_annulus',c_cnmf')))