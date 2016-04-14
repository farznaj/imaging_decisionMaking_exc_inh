% allows me (Kachi) to run files on my personal computer 11/16/2012

% function [all_data prop_correct_easy totalCompletedtrials] = plot_n_days_mice_kachimac(mousename, day, n, species, nlx_flag, plot_fit, which_modalities)
% % function [all_data] = plot_n_days(mousename, day, n, species, nlx_flag,
% % plot_fit, which_modalities)
% % NB -- which_modalities is an optional input argument for plotting subsets of the modalities.
% % it must be a cell array eof strings containing some subset of {'aud','vis','cc'};
% close all
% 
% if nargin < 6
%     plot_fit = 0;
% end
% 
% if nargin < 7
    which_modalities = {'cc','aud','vis'};
% end
% 
% if nargin < 5
%     nlx_flag= 0;
% end;
% 
% if nargin == 3
%     species = 'rat';
% end;


figure;
% h1 = axes('position',[0.1300    0.1100    0.3347    0.7150]); %This is for the CHOICE DATA
% hold on;
% 
% h2 = axes('position',[0.5703    0.1100    0.3347    0.7150]);  %THIS IS FOR THE RT
% hold on;



% for i=1:1
% 
%     this_is_the_day = i;
%     myday = datestr(datenum(day)-(i-1));
% 
%     if ismac
%         dataset = combine_mouse_sessions_ko(mousename, myday, n,species);
%     else
%         dataset = combine_days_mice_workinprogress(mousename, myday, n,species);
%     end
    
    all_data = [datastruct.filter_dataMat];
    myday = date;
    mousename = 'am008';
    
    
    
%     if nlx_flag
%         return
%     end;
    
    visual_trials = [all_data.visual_or_auditory];
    parsed_events = [all_data.parsed_events];
    wait_durations = [all_data.wait_duration];
    stimulus_strengths = [all_data.actual_stimulus_strength];
    catch_trial = [all_data.catch_trial];
    total_sound_duration = [all_data.total_sound_duration];
    
%     plot_title = sprintf('%s %s', dataset(1).rat, dataset(1).day);
%     disp(plot_title)
%     fprintf('number of trials = %d\n', length(visual_trials))
%     keyboard;
    
    indexes_va = find(visual_trials == 0); % visual & auditory
    indexes_a = find(visual_trials == -1); % only auditory
    indexes_v = find(visual_trials == 1); % only visual
    
    data = {};
    
    % visual & auditory
    data{1}.parsed_events = parsed_events(indexes_va);
    data{1}.stimulus_strengths = stimulus_strengths(indexes_va);
    data{1}.wait_durations = wait_durations(indexes_va);
    data{1}.catch_trial = catch_trial(indexes_va);
    data{1}.total_sound_duration = total_sound_duration(indexes_va);
    
    % only auditory
    data{2}.parsed_events = parsed_events(indexes_a);
    data{2}.stimulus_strengths = stimulus_strengths(indexes_a);
    data{2}.wait_durations = wait_durations(indexes_a);
    data{2}.catch_trial = catch_trial(indexes_a);
    data{2}.total_sound_duration = total_sound_duration(indexes_a);
    
    % only visual
    data{3}.parsed_events = parsed_events(indexes_v);
    data{3}.stimulus_strengths = stimulus_strengths(indexes_v);
    data{3}.wait_durations = wait_durations(indexes_v);
    data{3}.catch_trial = catch_trial(indexes_v);
    data{3}.total_sound_duration = total_sound_duration(indexes_v);
    
    % colors for plots: blue for vis+aud, green for only aud, black for only vis
    
    %sat_val = (i-1) * 0.4;
    %if sat_val >=1
    %    sat_val = 0.8;
    %end;
    sat_val=0;
    
    %color = [(0+sat_val) 1 (0+sat_val); 1 (0+sat_val) (0+sat_val); (0+sat_val) (0+sat_val) 1];
    color = [(0+sat_val) (0+sat_val) 1; (0+sat_val) 1 (0+sat_val); sat_val sat_val sat_val];
    % line style for plots
    line_style = {'-', '-', '-'};
    
    
    %     h3 = figure;
    %     hold on;
    
    completed_Trials_per_modality = [0 0 0];
    total_trials_per_modality = [0 0 0];
    
    all_sds = []; %we will put the fits from the cumulative gaussian for each condition in here.
    
    
    % In this loop, we go through the conditions: auditory, visual and multisensory.
    
    ConditionsToPlot = find([max(strcmp('cc',which_modalities)) max(strcmp('aud',which_modalities)) max(strcmp('vis',which_modalities))]);
    
    for j = ConditionsToPlot
        
        ss = unique(data{j}.stimulus_strengths); %stimulus strengths
        
        % define categories of stimulus strengths
        c1 = ss(ss>=0 & ss<0.05);
        c2 = ss(ss>=0.05 & ss<0.15);
        c3 = ss(ss>=0.15 & ss<0.25);
        c4 = ss(ss>=0.25 & ss<0.35);
        c5 = ss(ss>=0.35 & ss<0.45);
        c6 = ss(ss>=0.45 & ss<0.55);
        c7 = ss(ss>=0.55 & ss<0.65);
        c8 = ss(ss>=0.65 & ss<0.75);
        c9 = ss(ss>=0.75 & ss<0.85);
        c10 = ss(ss>=0.85 & ss<0.95);
        c11 = ss(ss>=0.95 & ss<=1);
        
        categories = {c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11};
        
        % now we need to get rid of the empty categories
        mean_categories = [];
        indexes_nec = []; % nec means non-empty-category
        for i = 1:length(categories)
            if ~isempty(categories{i})
                mean_categories = [mean_categories mean(categories{i})];
                indexes_nec = [indexes_nec i];
            end
        end
        % replace categories by non-empty-categories
        categories = {categories{indexes_nec}};
        
        trim_trials = 1; % number of trials we want to discard at beginning and end
        
        total = zeros(1,length(categories));
        correct = zeros(1,length(categories));
        
        % total_catch = zeros(2,length(categories));
        % correct_catch = zeros(2,length(categories));
        
        % preallocate memory
        wait_duration = zeros(length(data{j}.parsed_events) - trim_trials);
        
        for  i_trim = trim_trials : length(data{j}.parsed_events) - trim_trials
            
            total_trials_per_modality(j) = total_trials_per_modality(j) + 1;
            
            % histogram of RT
            if length(data{j}.parsed_events(i_trim).states.tone_playing) == 2
                wait_duration1 = data{j}.parsed_events(i_trim).states.tone_playing(2) - data{j}.parsed_events(i_trim).states.tone_playing(1);
            else
                wait_duration1 = 0;
            end
            wait_duration2 = 0;
            
            if (isempty(data{j}.parsed_events(i_trim).states.npunish))
                
                wait_duration2 = data{j}.parsed_events(i_trim).states.tone_playing2(2) - data{j}.parsed_events(i_trim).states.tone_playing2(1);
            end
            wait_duration(i_trim,:) = wait_duration1 + wait_duration2;
            
            % psychometric plot (performance)
            for k = 1:length(categories)
                if ismember(data{j}.stimulus_strengths(i_trim), categories{k})
                    
                    
                    if ~isempty(data{j}.parsed_events(i_trim).states.right_on) && data{j}.stimulus_strengths(i_trim) < 0.5
                        fprintf('Warning: may be misalignment because correct choice was right but stim strength was %0.03f', data{j}.stimulus_strengths(i_trim))
                    end
                    
                    
                    if (isempty(data{j}.parsed_events(i_trim).states.npunish))
                        total(k) = total(k) + 1;
                        completed_Trials_per_modality(j) = completed_Trials_per_modality(j) + 1;
                    end
                    if (~isempty(data{j}.parsed_events(i_trim).states.reward))
                        correct(k) = correct(k) + 1;
                    end
                    
                    
                end
                
            end
        end
        %I am adding some code so the program will output the number of
        %correct responses
        
        length_correct = length(correct);
        correct_for_sum(j,1:length_correct) = correct;   % computes correct responses for each modality, each row represents a modality and each column a stimulus strength
       
        % trim wait_duration vector
        wait_duration = wait_duration(trim_trials:length(wait_duration));
        
        % calculate performance
        performance = correct./total;
        %performance_catch = correct_catch./total_catch
        
      
        
        % to get the proportion of rightward choices we need to invert the performance for strengths < 0.5
        % we need this if we want to plot the unfolded psychometric curve
        prop_chose_right = performance;
        %prop_chose_right_catch = performance_catch;fprintg
        for t = 1:length(categories)
            if mean_categories(t) < 0.5
                prop_chose_right(t) = 0.5 - (performance(t) - 0.5);
                %prop_chose_right_catch(:,t) = 0.5 - (performance_catch(:,t) - 0.5);
                %collapsed_correct(t) =  correct(t) + correct(length(correct)+1-t);
                %collapsed_total(t) =  total(t) + total(length(total)+1-t);
            end
        end
        %collapsed_performance = collapsed_correct./collapsed_total;
        
        number_chose_right = prop_chose_right .* total;
        
%         axes(h1)
        % plot the data points
        plot(mean_categories, prop_chose_right, '-x', 'color', color(j,:));
        % catch trials
        % plot(mean_categories, prop_chose_right_catch(1,:), 'o', 'color', [1 0 1]);
        % plot(mean_categories, prop_chose_right_catch(2,:), '*', 'color', [1 0 1]);
        
        %if strcmp(mousename, 'ac001') || strcmp(mousename, 'ac002') || (strcmp(mousename, 'ac003') && j ~= 3) || (strcmp(mousename, 'ac004') && j == 2)
        % plot the fit
        if ~isempty(prop_chose_right)
         prop_correct_easy(j) = mean([(1-prop_chose_right(1)) prop_chose_right(end)]);
        end;
        
        if ~isempty(mean_categories)  %make sure we have data for this condition; we don't run each condition every day.
            axes(h1)
            
            
            if plot_fit
                [x, y, B, diag] = psych_fit_plot(mean_categories, number_chose_right, total);
                
                if isfield(B,'thetahat')  %this parameter changed name from one version of psignifit to the next
                    all_sds = [all_sds;B.thetahat(1:2)]; %this is mean and then variance.
                elseif isfield(B,'params_estimate')
                    all_sds = [all_sds;B.params_estimate(1:2)]; %this is mean and then variance.
                end;
                
                plot(x, y, 'color', color(j,:),'linewidth',3-(0.4*this_is_the_day));
                %end
               
            end
            
%             [a, b] = hist(wait_duration, 0:0.03:2);
%             axes(h2);
%             % plot the histogram
%             2-(j*0.3);
%             j;
%             
%             plot(b,a/sum(a), 'color', color(j,:), 'LineWidth', (2-(j*0.4)), 'LineStyle', line_style{j})
%             line([mean(data{j}.wait_durations) mean(data{j}.wait_durations)], [0 max(a/sum(a))], 'LineWidth', 1, 'LineStyle', '--', 'color', color(j,:));
            
            %diag.thetahat
            %B.thetahat
            
            %         figure(h3);
            %         % collapsed performance plot
            %         %plot(mean_categories(length(mean_categories)/2+1:length(mean_categories)), fliplr(collapsed_performance), 'o-', 'color', color(j,:));
            %
            %         bino_stderr = nan(1,length(collapsed_total));
            %         iterations = 10000; %this is how many coin clips you will have.
            %         for p = 1:length(collapsed_total)
            %             N = collapsed_total(p);
            %             bino_dist = binornd(N, 0.5, iterations, 1);
            %             bino_stderr(p) = std(bino_dist/N);
            %         end
            %         errorbar(mean_categories(length(mean_categories)/2+1:length(mean_categories)), fliplr(collapsed_performance), fliplr(bino_stderr), 'color', color(j,:));
        end; %is ~isempty
    end
    correct_trials_per_modality = sum(correct_for_sum,2);  %each row represents a modality and each column a stimulus strength
    totalCorrecttrials = sum(correct_trials_per_modality); %grand total of correct trials, across modalities
    
    totalCompletedtrials = sum(completed_Trials_per_modality);  %grand total number of trials counted as complete
%     keyboard
    
    water_amt = (totalCorrecttrials*4)/1000  %amount of water received

    totalTrials = sum(total_trials_per_modality); %all trials
    percent_complete = totalCompletedtrials/ totalTrials * 100;            %percentage of total trials that were complete
    percent_correct = totalCorrecttrials/totalCompletedtrials *100;  %  percentage of completed trials that were correct
%     keyboard
    
    percent_complete_per_modality = 100*completed_Trials_per_modality./total_trials_per_modality;
    percent_correct_per_modality = 100*(correct_trials_per_modality)'./completed_Trials_per_modality;
    
    fprintf('%d completed trials (all modalities), %6.2f%% of total trials \n', totalCompletedtrials,percent_complete);
    fprintf('%d correct trials, %6.2f%% of completed trials \n', totalCorrecttrials,percent_correct);    

    axes(h1)
    % performance plot style
    line([0 1], [0.5 0.5], 'LineStyle', '--', 'color', 'r');
    set(gca,'TickDir','out','Box','off','fontname','times','ticklength',[0.01 0],'ygrid','on');
    set(get(gca,'XLabel'),'String','Stim Strength','fontname','times');
    set(get(gca,'YLabel'),'String','Proportion rightward','fontname','times');
    set(get(gca,'Title'),'fontname','times');
    title([mousename ' performance ' myday ]);
    set(gca,'ylim',[0 1]);
    myy = get(gca,'ylim');
    ypos = 0.9*(myy(2));
    
    if plot_fit
        %sd_ind = 1;
        for i = 1:size(all_sds,1)
            text(0.1, ypos - (0.1 * (i-1)), num2str(all_sds(i,2)), 'color', color(ConditionsToPlot(i),:));
            %text(0.1, ypos-0.01, num2str(all_sds(2,2)), 'color', color(2,:));
            %text(0.1, ypos-0.02, num2str(all_sds(3,2)), 'color', color(3,:));
            %sd_ind = sd_ind + 1;
        end;
    end
    
    
    
%     axes(h2);
%     % histogram plot style
%     set(gca,'TickDir','out','Box','off','fontname','times','ticklength',[0.01 0],'ygrid','on');
%     set(get(gca,'XLabel'),'String','Wait duration (s)','fontname','times');
%     set(get(gca,'YLabel'),'String','Proportion of trials','fontname','times');
%     set(get(gca,'Title'),'fontname','times');
%     title([mousename ' Response Time ' myday]);
%     myy = get(gca,'ylim');
%     ypos = 0.9*(myy(2));
%     text(0.5, ypos+0.01, '% completed, %correct ', 'color', [1 0 0]);
% %     keyboard
%     sd_ind = 1;
%     for ii = 1:3
%         text(0.5, ypos - (0.01 * (ii-1)),[num2str(percent_complete_per_modality(ii),4) '%,' num2str(percent_correct_per_modality(ii),4) '%'], 'color', color(ii,:));
%         %text(0.05, ypos-0.05, num2str(percent_complete(2)), 'color', color(2,:));
%         %text(0.05, ypos-0.1, num2str(percent_complete(3)), 'color', color(3,:));
%     end;
    
    
    %     figure(h3);
    %     % collapsed performance plot
    %     set(gca,'TickDir','out','Box','off','fontname','times','ticklength',[0.01 0],'ygrid','on');
    %     set(get(gca,'XLabel'),'String','Stim Strength','fontname','times');
    %     set(get(gca,'YLabel'),'String','Proportion
    %     correct','fontname','times');
    %     set(get(gca,'Title'),'fontname','times');
    %     title(plot_title);
    
    
% end;

ImageName = [mousename '_results_' myday '.jpg'];
set(gcf,'paperposition',[0.25 6 5 5]);
%print -djpeg temp.jpg
%eval(sprintf('!mv temp.jpg %s_%s.jpg',mousename,day))

% if ismac
%    % keyboard;
%    ImageOutDir = '~/Churchland Lab/Dropbox/Kachi/Mice data plots';
%    ImageOutputDir = [ImageOutDir '/' myday '/'];
%    check =  exist(ImageOutputDir, 'dir');
%    if (check  == 0)
%        mkdir(ImageOutputDir);
%    end
%    
% %    print ('-djpeg','-r200',[ImageOutputDir '/' ImageName]); 
% %    print -djpeg ImageName;
% %     saveas(gcf,ImageName,'jpg');
% 
% end
% if ispc && exist('c:\ratter\ExperPort','dir')
%     cd 'c:\ratter\ExperPort'
% end

% saveimage(0,0, gcf, ImageName, myday)

% figure
% if n < 2
% % SummaryDataTable(myday, mousename, totalTrials,percent_complete,percent_correct)
% end


% end
% 
% function SummaryDataTable(myday, mousename, totaltrials,completedtrials,correcttrials)
% % close all 
% % clear all
% 
% f = figure('Position',[300 150 600 300]);
% 
% %default values
% rewardamt = 4;
% sessionduration = 0;
% % totaltrials = 0;
% % completedtrials = 0;
% % correcttrials = 0;
% rewardtype = 'ChooseSide2';
% PCPduration = 1;
% waitdurstep = 0.001;
% yogurt = false;
% enrichment = true; 
% 
% data1 = {rewardtype,PCPduration,waitdurstep,yogurt, enrichment};
% 
% rnames = []; 
% cnames = {'Reward Type    ','PCP duration/step','Wait duration/step', 'Yogurt drop  ', 'Enrichment'}; 
% 
% ceditable =[true true true true true]; 
% cformat = {{'Direct Reward' 'Allow Correction' 'ChooseSide1' 'ChooseSide2' 'ChooseSide3'}...
%     'numeric', 'numeric', 'logical','logical'};
%     
% addparam1 = uitable('ColumnName', cnames, 'RowName', rnames,...
% 'Position', [75 250 450 40],'Data', data1, 'ColumnEditable', ceditable);
% 
% data2 = {rewardamt, sessionduration,totaltrials,completedtrials,correcttrials};
% 
% rnames = []; 
% cnames = {'Reward Size (uL)','Session Time', 'Total Trials', 'Completed Trials % ', ...
%     'Correct Trials % ', }; 
% 
% ceditable =[true true true true true true true true true true]; 
% cformat = {'numeric', 'numeric','numeric', 'numeric', 'numeric',};
%     
% addparam2 = uitable('ColumnName', cnames, 'RowName', rnames,...
% 'Position', [75 200 450 40],'Data', data2, 'ColumnEditable', ceditable);
% 
% 
% % % stimulus modality table
% stimdata = zeros(6,3); %user enters manually
% stimcol = {'Visual' 'Auditory' 'Multisensory'};
% stimrow = {'0' '0.1' '0.2' '0.3' '0.4' '0.5'};
% stimedit = [true true true];
% stimformat = {'numeric' 'numeric' 'numeric'};
% stimulustable = uitable('ColumnName', stimcol,'Position', [165 65 270 125],...
%     'RowName', stimrow, 'ColumnEditable',stimedit,'ColumnFormat',stimformat,...
%     'Data', stimdata);
% 
% ImageName = [mousename '_table_' myday '.jpg'];
% 
% b = uicontrol('Style', 'pushbutton','String','SAVE!',...
%     'Position',[275 10 50 20], 'Callback',{@saveimage, gcf,ImageName,myday});
% 
% 
% end
% 
% function saveimage(src, eventdata, f,ImageName, myday)
% 
% % set(gcf,'paperposition',[0.25 6 5 5]); 
% 
% % ImageOutDir = '~/Churchland Lab/Dropbox/Kachi/Mice data plots';
%  ImageOutDir = '~/Desktop/';
%  
% ImageOutputDir = [ImageOutDir '/' myday '/'];
% check =  exist(ImageOutputDir, 'dir');
% if (check  == 0)
%     mkdir(ImageOutputDir);
% end
% 
% print (f, '-djpeg','-r200',[ImageOutputDir '/' ImageName]);
%    
% % saveas(gcf,SummaryTable,'jpg');
% 
% 
% 
% end
% 
% 
% 
% 
% 
