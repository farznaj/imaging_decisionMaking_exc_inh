function processCaImagingMCPnev(paramsFileName)
% processCaImagingMCPnev(paramsFileName)
%
% This is the master function for Matt's Ca imaging pathway. It performs
% the following steps:
% 1) Load a parameters file from a folder in the current directory named 
%    improcparams/paramsFileName. This file should be written by
%    writeCaProcessParams()
% 2) Motion correction using dftregistration. This is a fast, subpixel,
%    rigid-frame translation. The core function is preprocessCaMovies(). If
%    more than one channel is recorded, motion correction can be done on a
%    particular channel and the results can be applied to other channels.
%    The motion correction step will be skipped automatically if it has
%    already been performed. 
%    This step will write tif files ending in _MCM, and mat files named
%    "yymmdd_xxx_zzz", where yymmdd is the date that the mdf file was
%    recorded, xxx and zzz are the tif 'major' and 'minor' numbers,
%    respectively; these mat files include DFToutputs, ie the output of the
%    dftregistration, for each tif 'minor' file. For the entire movie (ie
%    all tif files corresponding to a mdf file) a mat file (named
%    "yymmdd_xxx") will be created which includes badFrames, pmtOffFrames,
%    DFTouputs, and maskBounds corresponding to the entire movie.
% 3) If requested, append to the "yymmdd_xxx" mat file the median, std dev,
%    max, and range images of the entire movie.
% 4) If requested, run Etychios and Liam's automatic source separation
%    algorithm. The core function is applyPnevPaninskiCaSourceSep(). The
%    results will be saved to a mat file named
%    [yymmdd_xxx-PnevPanResults-"nowStr"], where nowStr is the date and
%    time that the algorithm was run.
% 5) If requested, merge the results into the behavior file specified by
%    the supplied params file. This can optionally call a cleanup function
%    (from the params file) on the behavior data first. Saves yet another
%    file.
%
% To evaluate the output, load the result (located in the dataset's folder)
% and see ROIContoursPnev()



MCMSuffix = '_MCM';
% maxMaskWidth = 20;


%% Load parameters

loadVar = load(fullfile('improcparams', paramsFileName));
params = loadVar.params;


%{
%% Rename tif files so tif minor is XXX instead of XX.
% set current tif file names
if any(params.oldTifName)
    files = dir(params.tifFold);
    files = files(~[files.isdir]);

    % Parse all the filenames, save results into files struct array
    for f = 1:length(files)
      [nums, valid] = parseCaImagingTifName(files(f).name);
      files(f).nums = nums;
      files(f).valid = valid;
    end
    files = files([files.valid] & arrayfun(@(f) ismember(f.nums(2), params.tifNums(1,2)), files'));
    
    % rename tif files to the newest mview standard
    for ifile = 1: size(params.tifNums, 1)
        if params.oldTifName(ifile)
            fprintf('Renameing file %s\n', files(ifile).name)
            oldName = files(ifile).name;
            src = fullfile(params.tifFold, oldName);
            
            newName = assembleCaImagingTifName(params.tifNums(ifile,:), params.oldTifName(ifile));
            dest =  fullfile(params.tifFold, newName);
            
            movefile(src, dest)
        end
    end
end
%}


%% Figure out if preprocessing has been done for every tif file

preprocDone = true;
if ~params.motionCorrDone
    preprocDone = false;
end


% FN: the part below has issues and needs work. Even when MCM tif exists,
% the raw tif will make params.tifNums(f, 4) equal 0 and hence motion
% correction will be again performed! It is better and simpler to give
% motionCorrDone as an input to writeCaProcessParams, instead of
% having the script set it.
%{
nFiles = size(params.tifNums, 1);
for f = 1:nFiles
    if params.tifNums(f, 4) == 0
        fprintf('At least one file does not have a motion-corrected version. Performing motion correction.\n');
        preprocDone = false;
        break;
    end
    badFramesName = fullfile(params.tifFold, assembleCaImagingTifName(params.tifNums(f, :), 1));
    if ~exist(badFramesName, 'file')
        fprintf('At least one file does not have a _badFrames file. Performing motion correction.\n');
        preprocDone = false;
        break;
    end
end
%}


%% Do preprocessing if it hasn't been done already

date_major = sprintf('%06d_%03d', params.tifNums(1, 1:2));

if preprocDone
  fprintf('Motion correction has already been performed, skipping\n');

  %% Load tifs and set movieMC 
  if isempty(params.channelsToRead) % read all saved channels.
      chAll = unique(params.tifNums(:,4)); % channels saved.
      chAll(isnan(chAll)) = [];
  else
      chAll = params.channelsToRead;
  end
  
  movieMC = cell(1, max([chAll', params.dftRegCh, params.activityCh])); 
  for ch = chAll' 
      % Get list of MCM tif files corresponding to channel ch.
      tifNumsCh = params.tifNums(params.tifNums(:,4)==ch,:);
      tifList = cell(1, size(tifNumsCh,1));
      for f = 1:length(tifList)
          tifList{f} = fullfile(params.tifFold, assembleCaImagingTifName(tifNumsCh(f, :), params.oldTifName(f))); 
      end
      
%       movieMC{ch} = readTiffSet(tifList);
      for t = 1:length(tifList)
          fprintf('Reading tif file %s\n', tifList{t})
          movieMC{ch} = cat(3, movieMC{ch}, bigread2(tifList{t}));
      end
  end
  
  
  %% Set badFrames and pmtOffFrames
  if params.allTifMinors % analyze all tif minors.
      % Load badFrames and pmtOffFrames
      load(fullfile(params.tifFold, date_major), 'badFrames', 'pmtOffFrames')
      
      if ~exist('pmtOffFrames', 'var') % a = whos('-file', '151021_001.mat'); ~any(strcmp('pmtOffFrames', {a.name}))
          pmtOffFrames = cell(size(badFrames));
          for ch = 1:length(pmtOffFrames)
              pmtOffFrames{ch} = false(length(badFrames{ch}),1);
          end
      end
      
      
  else % only analyze some tifMinors.
      % load 'badFramesTif', 'pmtOffFramesTif' for each tifMinor, and
      % concatenate them to set 'badFrames', 'pmtOffFrames'.
      badFrames = cell(1, max(chAll));
      pmtOffFrames = cell(1, max(chAll));
      
      tifMinor = unique(params.tifNums(:,3))';
      for itm = 1:length(tifMinor)
          a = dir(fullfile(params.tifFold, [date_major, '_*', num2str(tifMinor(itm)), '.mat']));
          load(fullfile(params.tifFold, a.name), 'badFramesTif', 'pmtOffFramesTif')
                    
          if exist('badFramesTif', 'var')
              for ch = 1:length(badFramesTif)
                  badFrames{ch} = [badFrames{ch}; badFramesTif{ch}];
                  pmtOffFrames{ch} = [pmtOffFrames{ch}; pmtOffFramesTif{ch}];
              end
          end
      end      
      
      
      if ~exist('badFramesTif', 'var') % for some days badFramesTif and pmtOffFramesTif were not saved, so you need to set them here.
          % load badFrames and pmtOffFrames corresponding to the entire mdf file.
          load(fullfile(params.tifFold, date_major), 'badFrames', 'pmtOffFrames')
          
          if ~exist('pmtOffFrames', 'var') % a = whos('-file', '151021_001.mat'); ~any(strcmp('pmtOffFrames', {a.name}))
              pmtOffFrames = cell(size(badFrames));
              for ch = 1:length(pmtOffFrames)
                  pmtOffFrames{ch} = false(length(badFrames{ch}),1);
              end
          end
          
          % load DFToutputs of the first tif minor, and use its size to figure
          % out how many frames were saved per Tif file (except for the last
          % tif file that includes whatever frame is remained).
          a = dir(fullfile(params.tifFold, [date_major, '_*01.mat']));
          load(fullfile(params.tifFold, a.name), 'DFToutputs')
          nFramesPerMovie_est = size(DFToutputs{find(cellfun(@(x) ~isempty(x), DFToutputs),1)} , 1);
          
          % set the frames corresponding to tifMinor
          cs = [0:nFramesPerMovie_est:size(badFrames{1},1) length(badFrames{1})]; % [0 cumsum(nFramesPerMovie)];
          frs = [];
          for itm = tifMinor
              frames = cs(itm)+1 : cs(itm+1);
              frs = [frs , frames];
          end
          
          % extract those frames from badFrames and pmtOffFrames
          for ch = 1:length(pmtOffFrames)
              badFrames{ch} = badFrames{ch}(frs);
              pmtOffFrames{ch} = pmtOffFrames{ch}(frs);
          end
      end
  end

  
%%  
else 
    
  % TO-DO: Add handling for other-channel motion correction code here
  fprintf('Performing motion correction\n');
  
  % Get list of non-MCM tif files.
  tifNumsNoMCM = params.tifNums(isnan(params.tifNums(:,4)),:);
  tifList = cell(1, size(tifNumsNoMCM,1));
  for f = 1:length(tifList)
      tifList{f} = fullfile(params.tifFold, assembleCaImagingTifName(tifNumsNoMCM(f, :), params.oldTifName(f)));  % tifList does not include any _MCM tif files.
  end
  
  regTif = params.regTifFile;
  regFrameNums = params.regFrameNums;
  dftRegChannel = params.dftRegCh;
  channels2write = params.channelsToWrite;
  maxMaskWidth = params.maxMaskWidth;
  analysisDir = params.analysisFold;
  
  [movieMC, badFrames, pmtOffFrames] = preprocessCaMovies(tifList, regTif, regFrameNums, dftRegChannel, channels2write, MCMSuffix, maxMaskWidth, analysisDir, params.pmtOffThr);
  
end


[imHeight, imWidth, nFrames] = size(movieMC{params.activityCh});
save(fullfile(params.tifFold, date_major), 'params', 'imHeight', 'imWidth', '-append') % This mat file is already created during motion correction, which is why we append to it here.


%% Convert movieMC to double
% OK to convert to double here, because applyPnevPaninskiCaSourceSep checks
% to see whether it's already a double and will skip conversion if it is.
% Also std requires single or double inputs.
if ~isa(movieMC{params.activityCh}, 'double') && (params.pnevActivity || params.saveGoodMovieStats) 
  fprintf('Converting to double...\n');
  tic;  
  for ch = 1:length(movieMC)
      movieMC{ch} = double(movieMC{ch});
  end
  
  fprintf('%0.2fs\n', toc);
end


%% Produce and save averaged images: Median, std dev, max, range
 
if params.saveGoodMovieStats
  fprintf('Generating various projection images...\n');
  tic;
  
  % Pre-allocate cell arrays
  medImage = cell(1,length(movieMC));
  sdImage = cell(1,length(movieMC));
  maxImage = cell(1,length(movieMC));
  rangeImage = cell(1,length(movieMC));
  
  for ch = 1:length(movieMC)
    if ~isempty(movieMC{ch})
      % Pre-allocate
      medImage{ch} = NaN(imHeight, imWidth);
      sdImage{ch} = NaN(imHeight, imWidth);
      maxImage{ch} = NaN(imHeight, imWidth);
      rangeImage{ch} = NaN(imHeight, imWidth);

      % Compute images pixel by pixel. We do this so that we can skip bad frames
      % without having to allocate another entire movie.
      for i1 = 1:imHeight
        for i2 = 1:imWidth
          pixel = movieMC{ch}(i1, i2, ~badFrames{ch} & ~pmtOffFrames{ch});
          
          medImage{ch}(i1, i2) = median(pixel);
          sdImage{ch}(i1, i2) = std(pixel);
          maxImage{ch}(i1, i2) = max(pixel);
          rangeImage{ch}(i1, i2) = range(pixel);
        end
      end
    end
  end
  clear pixel
  fprintf('%0.2fs\n', toc);
  
  % Save results    
  save(fullfile(params.tifFold, date_major), 'medImage', 'sdImage', 'maxImage', 'rangeImage', '-append')
%   save(fullfile(params.tifFold, ['PnevPanResults-' nowStr]), ...
%       'imHeight', 'imWidth', 'medImage', 'sdImage', 'maxImage', 'rangeImage');
  
end


%% Run Eftychios and Liam's algorithm

if params.pnevActivity
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    
    % The following vars may be user defined:
    pnev_inputParams.K = params.numComps;                                % number of components to be found
    pnev_inputParams.temp_sub = params.tempSub;                          % temporal subsampling for greedy initiation, set to 1 for no down sampling.
    pnev_inputParams.space_sub = params.spaceSub;                        % spatial subsampling for greedy initiation, set to 1 for no down sampling.
    pnev_inputParams.finalRoundMCMC = params.finalRoundMCMC;             % do a final round of MCMC method (if false, after merging 2 iterations of const foopsi will be done. If true, after merging 1 iter of const foopsi and 1 iter of MCMC will be done.)
    pnev_inputParams.doPlots = params.doPlots;                           % if true, some figures and a movie will be made.
%     pnev_inputParams.parallelTempUpdate = params.parallelTempUpdate;     % do parallel temporal updating.
    pnev_inputParams.save4debug = params.save4debug;                     % save Eftychios's variables (eg A,C,etc) after each step for debug purposes. 
    pnev_inputParams.MCMC_B = params.MCMC_B;                             % number of burn in samples. eg. 200
    pnev_inputParams.MCMC_Nsamples = params.MCMC_Nsamples;               % number of samples after burn in. eg. 200
    pnev_inputParams.MCMC_prec = params.MCMC_prec;                       % specifies the extent of discarding the long slowly decaying tales of the ca response. eg. 1e-2
    pnev_inputParams.save_merging_vars = params.save_merging_vars;       % whether to save some vars related to merging components.
    pnev_inputParams.search_dist = params.search_dist;                   % search distance when updating spatial components.
    pnev_inputParams.concatTempUpdate = params.concatTempUpdate;         % if true, temporal updating will be done on the movie that includes concatenated trials (datasets), otherwise the trace of each dateset (trial) will be updated separately; this gives better estimate of time constants and removes artificial spikes at the begining of each dateset (trial).  

    if ~pnev_inputParams.concatTempUpdate % you need framesPerTrial already saved!
        if params.allTifMinors % analyze all tif minors.
            load(fullfile(params.tifFold, date_major), 'badFrames', 'framesPerTrial')
            numRecFrs = length(badFrames{1});

            framesPerTrialNoNaN = framesPerTrial(~isnan(framesPerTrial));
            cs_frtrs = unique([0 cumsum(framesPerTrialNoNaN) numRecFrs]); % if some frames were recorded without a trial, add those frames too.
        else
            error('FN: copy the codes from framesPerTrialMovie.m')
        end
    else
        cs_frtrs = [];
    end

    % The following vars rarely need a different value:
    pnev_inputParams.tau = 4;                                           % std of gaussian kernel (size of neuron) 
    pnev_inputParams.p = 2;                                             % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
    pnev_inputParams.merge_thr = 0.8;                                   % merging threshold
    pnev_inputParams.deconv_method = 'constrained_foopsi';              % activity deconvolution method
    pnev_inputParams.temp_iter = 2;                                     % number of block-coordinate descent steps 
    pnev_inputParams.fudge_factor = .98;                                % bias correction for AR coefficients
    
    
    [A, C, S, C_df, S_df, Df, b, f, srt, Ain, options, P, merging_vars] = demo_script_modif(movieMC{params.activityCh}, pnev_inputParams, cs_frtrs);
    
    save(fullfile(params.tifFold, [date_major, '_ch', num2str(params.activityCh), '-PnevPanResults-', nowStr]), ...
        'A', 'C', 'S', 'C_df', 'S_df', 'Df', 'b', 'f', 'srt', 'Ain', 'options', 'P', 'pnev_inputParams', 'merging_vars');
    
    activity = C;
    
    %{
    [spatialBasis, activity, spatialBackground, backgroundActivity, spiking, Df, activityDf, spikingDf, pnevParam, ROIOrderMeasure, greedySpatialBasis] = ...
        applyPnevPaninskiCaSourceSep(movieMC{params.activityCh});
%     [A, C, b, f, S, Df, C_df, S_df, P, ROIOrderMeasure, Ain]

    save(fullfile(params.tifFold, [date_major, '-PnevPanResults-', nowStr]), ...
        'spatialBasis', 'activity', 'spatialBackground', 'backgroundActivity', ...
        'spiking', 'Df', 'activityDf', 'spikingDf', 'pnevParam', 'ROIOrderMeasure', 'greedySpatialBasis');
    %}
end


%% If requested, merge into behavior

if ~isempty(params.behavFile)
  
  %% Get trialization info
  
  framesPerTrial = cell(1, length(params.binFiles));
  trialNumbers = cell(1, length(params.binFiles));
  frame1RelToStartOff = cell(1, length(params.binFiles));
  
  for f = 1:length(params.binFiles)
    [framesPerTrial{f}, trialNumbers{f}, frame1RelToStartOff{f}] = ...
      framesPerTrialStopStart3An(params.binFiles{f}, params.framecountFiles{f}, params.headerBug);
  end
  
  framesPerTrial = [framesPerTrial{:}];
  trialNumbers = [trialNumbers{:}];
  frame1RelToStartOff = [frame1RelToStartOff{:}];
  
  
  %% Load the alldata file
  
  loadVar = load(params.behavFile);
  alldata = loadVar.all_data;
  
  
  %% If specified, call cleanup function
  
  if ~isempty(params.behavFcn)
    alldata = feval(params.behavFcn, alldata);
  end

  
  %% Merge activity into alldata, save
  
  ad = mergeActivityIntoAlldataPnev(alldata, C_df(1:end-1, :)', S_df', framesPerTrial, trialNumbers, frame1RelToStartOff, badFrames{params.activityCh}, pmtOffFrames{params.activityCh});
%   ad = mergeActivityIntoAlldataPnev(alldata, activityDf', spikingDf', framesPerTrial, trialNumbers, frame1RelToStartOff, badFrames{params.activityCh}, pmtOffFrames{params.activityCh});
  save(params.mergedName, 'ad');

end


fprintf('Done.\n');
