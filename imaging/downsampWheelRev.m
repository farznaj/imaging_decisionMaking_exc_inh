function [speed_d_allMice, wheelRev_d_allMice, td_allMice, nPreMin_wheelRev_allMice_ds] = downsampWheelRev(wheelRev_aligned_allMice, time_aligned_wheelRev_allMice, nPreMin_wheelRev_allMice, downSampBin, binLength, mice)
%% Set speed (units: mm/sec) of mouse running on the wheel ... compute it in 100ms bins (ie every 10 bin)
% first downsample the wheelRev trace, then compute speed on it.

nPreMin_wheelRev_allMice_ds = nan(1, length(mice));
td_allMice = cell(1,length(mice));
speed_d_allMice = cell(1,length(mice));
wheelRev_d_allMice = cell(1,length(mice));

for im = 1:length(mice)
    
    %%%%%%% Downsample the wheelRev trace
    
    tb = nPreMin_wheelRev_allMice(im)+1: -downSampBin : -downSampBin; % tb(i) : tb(i+1)-1
    tb = tb(end:-1:1); tb(tb<0) = [];
%     e = nPreMin_wheelRev_allMice{im};
%     f = e - downSampBin*floor(e/downSampBin) + 1 : e;    
    x = wheelRev_aligned_allMice{im}(:,tb(1): tb(end));
    tt = floor(size(x,2) / downSampBin); % number of time points in the downsampled X including frames after frame0
    x = x(:, 1: downSampBin*tt);
    xx = reshape(x, [size(x,1), downSampBin, tt]); % nDays x downSampBin x tt
    xdb = squeeze(mean(xx, 2)); % nDays x tt % downsampled trace including bins after nPreMin
    
    nPreMin_wheelRev_allMice_ds(im) = size(xdb,2);
    
    
    ta = nPreMin_wheelRev_allMice(im)+1 : downSampBin :length(time_aligned_wheelRev_allMice{im});        % ta(i) : ta(i+1)-1        
%     eventI = nPreMin_wheelRev_allMice{im}+1;
%     lenPost = length(time_aligned_wheelRev_allMice{im}) - (eventI-1);
%     f = eventI : (eventI-1) + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins        
    x = wheelRev_aligned_allMice{im}(:,ta(1): end);
    tt = floor(size(x,2) / downSampBin); % number of time points in the downsampled X including frames after frame0
    x = x(:, 1: downSampBin*tt);
    xx = reshape(x, [size(x,1), downSampBin, tt]); % nDays x downSampBin x tt
    xda = squeeze(mean(xx, 2)); % nDays x tt % downsampled trace including bins after nPreMin
    
    % set the final downsampled trace
    xd = cat(2, xdb, xda);    

    
    %%% set downsampled time
    tdb = sort(-binLength * (1:nPreMin_wheelRev_allMice_ds(im)-1));
    tda = sort(binLength * (0:size(xd,2)-nPreMin_wheelRev_allMice_ds(im)));
    td = cat(2,tdb,tda) * downSampBin;    

%     figure; plot(td, mean(xd,1)); 
%     figure; plot(time_aligned_wheelRev_allMice{im}, mean(wheelRev_aligned_allMice{im},1))    
    
    
    %%%% set the speed trace
    speed_d = diff(xd,1,2); % In the difference trace: the 1st nPreMin_wheelRev_allMice_ds(im)-1 elements are before choice; nPreMin_wheelRev_allMice_ds(im) is the 1st element after choice.
    speed_d = speed_d / (downSampBin * binLength / 1000); % convert to mm / sec
    
%     figure; hold on; 
%     plot(td(1:end-1), speed_d'); 
%     plot(td(1:end-1), mean(speed_d,1), 'k', 'linewidth',2)
%     vline(0) %nPreMin_wheelRev_allMice_ds(im)
    
    
    %%%% keep vars of all mice
    wheelRev_d_allMice{im} = xd;
    td_allMice{im} = td;
    speed_d_allMice{im} = speed_d;

end
