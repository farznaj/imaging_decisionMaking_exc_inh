%This was adapted by AKC from code written by JS on 11-24-2012. 
%It returns the success and failure biases as well as the regular offset. 
%Usage: [bs bf b0] = mouse_prob_choice('am014',3,'26-Nov-2012')
%Slight modification of input arguments - Kachi O 13-Feb-2013

function [bs bf b0 mousename numtrials] = mouse_prob_choice_file(ratname, day, n_days_back, mod)
% function [bs bf b0] = mouse_prob_choice_kachi(mousename,ndays,theDate, fig)


if nargin < 2
% mousename = 'am008';
% ndays = 2;
fig = 1; % 1 = plot figure 0= do not plot

end

% if nargin < 3
% mousename = 'am008';
% ndays = 2;
% theDate = '01-Feb-2013';
% fig = 1;
% end

% if nargin < 4
% fig = 1;
% end

% dat1 = combine_rat_sessions_ko(mousename,theDate,ndays,'mice');
% dat1 = dat1.data;

% cd '~/Churchland Lab/repoland/playgrounds/Kachi/data/'
% cd 'C:\Users\fnajafi\Dropbox\ChurchlandLab\data for trial history'

% ratname = 'am053';
% folder = fullfile('Z:','data', ratname, 'behavior');
% cd(folder)

% dat1 = 
% load(filename);

[data, days_loaded] = ab_loadRatBehavioralData_vSNR(ratname, day, n_days_back);

dat1 = data; % all_data;
% dat1(end) = [];

% keyboard
% mousename = dat1.subject;
% i = regexp(filename, 'am'); 
mousename = ratname; % 'am053'; % filename(i:i+4);
% dat1 = dat1.DataMatx;


if isempty(dat1)
sprintf('no data')
bs=[NaN NaN]; bf=[NaN NaN]; b0=[NaN NaN];
    return
end;
% close;

dat2 = [];
dat = dat1;

if mod == 1
    %mod = 1 remove auditory trials
%     dat([dat.visual_or_auditory] == -1) =[];  %remove auditory trials (temp)
    dat([dat.visualOrAuditory] == -1) =[];  %remove auditory trials (temp)
end



 

if ~isfield(dat,'this_vis_click_rate')
   addClickRates; %Add auditory to this!!  Done by AKC super fast because I can't figure out where john normally assigns this_vis_click_rates. 
end;



% first we need trial types
ss_low_rel_aud_trials = [dat.show_audio] & ~[dat.show_visual];% & ([dat.this_aud_SNR] == 1); %this is commented out because we only have one noise level for now
%ss_high_rel_aud_trials = [dat.show_audio] & ~[dat.show_visual] & ([dat.this_aud_SNR] == 2);
ss_vis_trials = [dat.show_visual] & ~[dat.show_audio];
cc_low_rel_aud_vis_trials = [dat.show_audio] & [dat.show_visual] ;%& ([dat.this_aud_SNR] == 1);
%cc_high_rel_aud_vis_trials = [dat.show_audio] & [dat.show_visual] ;%& ([dat.this_aud_SNR] == 2);

modality = [dat.visualOrAuditory];

%ss_vis_trials = (modality == 1);  %-1 is auditory; 0 is MS, 1 is visual. 
%ss_ms_trials = (modality == 0);
%ss_high_rel_aud_trials = (modality == -1); %actually these aren't really high reliability buyt anyway. 



%You will have to change this to add the auditory back in! 
trial_type_matrix = [ss_vis_trials(:) ss_low_rel_aud_trials(:) cc_low_rel_aud_vis_trials(:)];%[ss_low_rel_aud_trials(:), ss_high_rel_aud_trials(:), ss_vis_trials(:), cc_low_rel_aud_vis_trials(:), cc_high_rel_aud_vis_trials(:)];


% now we need trial histories for strategy parameters
disp('Computing trial histories (successes/failures)')

[dat.last_trial_from_same_session] = deal(NaN);
% keyboard;

for jj = 2:length(dat)
dat(jj).last_trial_from_same_session = 1; % - diff([dat(jj-1:jj).session_count]); % FN: important!!
end


trial_success_history = zeros(1,length(dat)); trial_failure_history = zeros(1,length(dat));

%the "correct_choice" was an unfortunate name for that structure field. "correct_choice" is actually referring to the rewarded side on the given trial, not whether the animal made a correct choice.
%i.e. correct_choice = 0 means rate < CB (should go left) and correct_choice= 1 means rate > CB (should go right).
%So trial_success_history is looking for trials where the preceding trial was slow rate and the animal went left (-1) or high rate and the animal correctly went right (1) -- vice versa for trial_failure_history
trial_success_history(2:end) = ([dat(2:end).last_trial_from_same_session] == 1) .* (-1*([dat(1:end-1).resp] == 1 & [dat(1:end-1).correctSide] == 0) + 1*([dat(1:end-1).resp] == 2 & [dat(1:end-1).correctSide] == 1));
trial_failure_history(2:end) = ([dat(2:end).last_trial_from_same_session] == 1) .* (-1*([dat(1:end-1).resp] == 1 & [dat(1:end-1).correctSide] == 1) + 1*([dat(1:end-1).resp] == 2 & [dat(1:end-1).correctSide] == 0));

%I wrote a little script called, "checkSuccessFail" that double checks that
%this is working. 


%Occasionally we have days where there are just almost no non-aborted
%trials. This makes estimates of trial history effects laughable. 
size(find(trial_failure_history ~=0),2);


if size(find(trial_failure_history ~=0),2) < 20 | size(find(trial_success_history ~=0),2) < 40
    
    
    sprintf('fewer than 10 percent non-aborted successes of failures; insufficient data to compute trial history\n')
    bs=[NaN NaN]; bf=[NaN NaN]; b0=[NaN NaN];
    return
end;


% now we need the distance from category boundary for each type of stimulus.
disp('Computing rate distances from CB');

category_boundary_rate = 12.5;
vis_dist_from_category_boundary_rate = [dat.this_vis_click_rate] - category_boundary_rate;
aud_dist_from_category_boundary_rate = [dat.this_aud_click_rate] - category_boundary_rate;

ss_low_rel_aud_dist_from_category_boundary_rate = ss_low_rel_aud_trials .* aud_dist_from_category_boundary_rate;
%ss_high_rel_aud_dist_from_category_boundary_rate = ss_high_rel_aud_trials .* vis_dist_from_category_boundary_rate;
ss_vis_dist_from_category_boundary_rate = ss_vis_trials .* vis_dist_from_category_boundary_rate;
cc_low_rel_aud_vis_aud_dist_from_category_boundary_rate = cc_low_rel_aud_vis_trials .* aud_dist_from_category_boundary_rate;

%Note that if you ever do the independent multisensory stimuli, you will
%need to put this back in!
%cc_low_rel_aud_vis_vis_dist_from_category_boundary_rate = cc_low_rel_aud_vis_trials .* vis_dist_from_category_boundary_rate;

%cc_high_rel_aud_vis_aud_dist_from_category_boundary_rate = cc_high_rel_aud_vis_trials .* aud_dist_from_category_boundary_rate;
%cc_high_rel_aud_vis_vis_dist_from_category_boundary_rate = cc_high_rel_aud_vis_trials .* vis_dist_from_category_boundary_rate;


%all_conditions_rate_dist_mat = ss_vis_dist_from_category_boundary_rate(:);

%For John this was 6 because he has 2 more multisensroy conditions. 
all_conditions_rate_dist_mat = [ss_low_rel_aud_dist_from_category_boundary_rate(:), ss_vis_dist_from_category_boundary_rate(:), ... 
                                cc_low_rel_aud_vis_aud_dist_from_category_boundary_rate(:)];%    %Will need this for independent condition! :   , cc_low_rel_aud_vis_vis_dist_from_category_boundary_rate(:) ];

% construct choice matrix
disp('constructing choice matrix');
choice_data_matrix = zeros(length(dat),10);


choice_data_matrix(:,1:3) = [ss_low_rel_aud_trials(:), ss_vis_trials(:), cc_low_rel_aud_vis_trials(:)];
choice_data_matrix(:,8:10) = all_conditions_rate_dist_mat;
clear all_conditions_rate_dist_mat cc*dist*rate

%From John:
%choice_data_matrix(:,1:5) = [ss_low_rel_aud_trials(:), ss_high_rel_aud_trials(:), ss_vis_trials(:), cc_low_rel_aud_vis_trials(:), cc_high_rel_aud_vis_trials(:)];
%choice_data_matrix(:,18:24) = all_conditions_rate_dist_mat;


past_trial_type_matrix = [zeros(1,size(trial_type_matrix,2)); trial_type_matrix(1:end-1,:)];



%START HERE.
each_ss_cond_trial_success_history = repmat(trial_success_history(:),1,2) .* past_trial_type_matrix(:,1:2);
each_ss_cond_trial_failure_history = repmat(trial_failure_history(:),1,2) .* past_trial_type_matrix(:,1:2);

%choice_data_matrix(:,3)  = trial_success_history(:) ;
%choice_data_matrix(:,4)  = trial_failure_history(:) ;

choice_data_matrix(:,4:5) = each_ss_cond_trial_success_history;
clear each_ss_cond_trial_success_history
choice_data_matrix(:,6:7) = each_ss_cond_trial_failure_history;
clear each_ss_cond_trial_failure_history


% now do the modality effects analysis here

valid_trials = [dat.resp] ~= 999;
valid_trials=valid_trials(:);
past_valid_trials = [0; valid_trials(1:end-1)];


%ss_aud_trials = ss_low_rel_aud_trials(:) | ss_high_rel_aud_trials(:);

ss_modality_matrix = [ ss_low_rel_aud_trials(:), ss_vis_trials(:)];
past_ss_modality_matrix = [zeros(1,size(ss_modality_matrix,2)); ss_modality_matrix(1:end-1,:)];

%ss_rate_dist_mat = [ss_vis_dist_from_category_boundary_rate(:)];

ss_rate_dist_mat = [ss_low_rel_aud_dist_from_category_boundary_rate(:),  ss_vis_dist_from_category_boundary_rate(:)];


% Can change line below to try only past valid trials vs all past trials
% (remove the past_valid_trials condition to change):

numconds = 3;

%AKC: I think this should now be a vector and it should just be the
%distance from the cat boundary for all visual trials. Why are there some
%trials that are not visual trials? Wait, I don't want this; I don't care
%about the modality effect because I have only one modality. 

%This is what I made at some point but I think it is wrong. 
%ss_rate_dist_from_cb_at_each_past_ss_modality = repmat(past_valid_trials(:),1,numconds) .* repmat(past_ss_modality_matrix,1,2) .* [repmat(ss_rate_dist_mat(:,1),1,2)];

ss_rate_dist_from_cb_at_each_past_ss_modality = repmat(past_valid_trials(:),1,4) .* repmat(past_ss_modality_matrix,1,2) .* [repmat(ss_rate_dist_mat(:,1),1,2), repmat(ss_rate_dist_mat(:,2),1,2)];
%ss_rate_dist_from_cb_at_each_past_ss_modality = repmat(past_valid_trials(:),1,6) .* repmat(past_ss_modality_matrix,1,3) .* [repmat(ss_rate_dist_mat(:,1),1,2), repmat(ss_rate_dist_mat(:,2),1,2), repmat(ss_rate_dist_mat(:,3),1,2)];

%choice_data_matrix(:,12:17) = ss_rate_dist_from_cb_at_each_past_ss_modality;
%clear ss_rate_dist_from_cb_at_each_past_ss_modality

response_vec = [dat.resp];
response_vec(find(response_vec == 999)) = NaN;
response_vec = response_vec - 1;

disp('fitting model with glmfit...');

Y = response_vec(:);
offset = zeros(size(Y));
pwts = ones(size(Y));
% keyboard;

choice_data_matrix(isnan(choice_data_matrix)) = 0;

%single_sensory_choice_matrix = choice_data_matrix;
%single_sensory_choice_matrix(find(nansum(choice_data_matrix(:,1:3),2) > 1),:) = 0;
%Y_ss = Y; offset_ss = offset; pwts_ss = pwts;
%Y_ss((find(nansum(choice_data_matrix(:,1:3),2) > 1))) = NaN;
%pwts_ss((find(nansum(choice_data_matrix(:,1:3),2) > 1))) = 0;

% fit prob choice model here:
%[coefficient_vec,model_deviance,model_stats] = glmfit(choice_data_matrix,Y,'binomial','logit','offset',offset,pwts,'constant','off');

%choice_data_matrix = [choice_data_matrix ones(length(choice_data_matrix),1)];
%Currently this is [isItVisual Rate Success Failure Constant]  %Is adding a
%constant the right thing to do??? 


numtrials = length(Y); 

[coefficient_vec,model_deviance,model_stats] = glmfit(choice_data_matrix,Y,'binomial','offset',offset,'constant','off');

Z_vals = choice_data_matrix * coefficient_vec;
choice_model_response_predictions = 1./(1+exp(-1*Z_vals));

%OutPath = [OutDir '/' subject '_reduced_strategy_and_modality_effect_prob_choice_model_data.mat'];
%delete(OutPath);
%save(OutPath,'coefficient_vec','model_deviance','model_stats','Z_vals','choice_model_response_predictions');


%So I guess coefficients 4 and 6 are what you want?
%[isItAud isItVis isItMS  success1 success2 failure1 failure2   audRate visRate msRate]


%keyboard;
% if mod == 1
%     makeModelPmfAllConds;    %plots all modalites
% else
    makeModelPmfAllConds_ko;    %plots only visual
% end


bs = [coefficient_vec(4) model_stats.se(4)];
bf = [coefficient_vec(6) model_stats.se(6)];
b0 = coefficient_vec(1:3);

return



%% PLOTTING SECTION!!

%{
conditions = {'vis'};

cmap = [32 255 32; 0 100 0; 0 0 0; 32 32 255; 0 0 100]/255;

fignum=0;
% now compute the choices in terms of past trial modalities

for ConditionCount = 1%:9
    
    this_cond_trials = find(ss_rate_dist_from_cb_at_each_past_ss_trial_type(:,ConditionCount));
    scores_this_cond = (1*([dat(this_cond_trials).resp] == 1 & [dat(this_cond_trials).correct_choice] == 0) + 1*([dat(this_cond_trials).resp] == 2 & [dat(this_cond_trials).correct_choice] == 1));
    scores_this_cond(find([dat(this_cond_trials).resp] == 999)) = NaN;
    
    modality_effects_prop_correct_ests(ConditionCount) = nanmean(scores_this_cond);
    modality_effects_prop_correct_stderrs(ConditionCount) = sqrt( modality_effects_prop_correct_ests(ConditionCount).*(1-modality_effects_prop_correct_ests(ConditionCount))./ numel(scores_this_cond(~isnan(scores_this_cond))) );
    
end

fignum = fignum + 1;
figure(fignum); clf(fignum);
hold on;

conditions = {'low rel aud','high rel aud','vis','cc low rel aud','cc high rel aud'};
cmap = [32 255 32; 0 100 0; 0 0 0; 32 32 255; 0 0 100]/255;

%for CurrentConditionCount=1:3
%    current_condition = conditions{CurrentConditionCount};
%end

X = sort(get_errorbar_xcoords_for_grouped_bar_plot(ones(3,3)));

for PastConditionCount = 1:3
    past_condition = conditions{PastConditionCount};
    legend_labels{PastConditionCount} = past_condition;
    this_past_cond_ests = modality_effects_prop_correct_ests(PastConditionCount:3:end);
    this_past_cond_stderrs = modality_effects_prop_correct_stderrs(PastConditionCount:3:end);
    this_past_cond_xcoord = X(PastConditionCount:3:end);
    this_past_cond_color = cmap(PastConditionCount,:);
    
    plot(this_past_cond_xcoord,this_past_cond_ests,'o','MarkerSize',8,'MarkerFaceColor',this_past_cond_color,'MarkerEdgeColor',this_past_cond_color);
    for jj = 1:numel(this_past_cond_xcoord)
        this_plot = plot([this_past_cond_xcoord(jj), this_past_cond_xcoord(jj)],[this_past_cond_ests(jj)-1.96*this_past_cond_stderrs(jj), this_past_cond_ests(jj)+1.96*this_past_cond_stderrs(jj)],'-','LineWidth',1.5,'Color',this_past_cond_color);
        set(get(get(this_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
end

this_legend = legend(legend_labels,'fontname','times','fontsize',16,'Location','Best');
set(this_legend,'Box','off');
this_legend_children = get(this_legend,'Children');
this_legend_text = this_legend_children(strcmp(get(this_legend_children,'Type'),'text'));  %# Select the legend children of type 'text'
set(this_legend_text,{'Color'},{cmap(3,:); cmap(2,:); cmap(1,:)});    %# Set the colors
this_legend_lines = this_legend_children(strcmp(get(this_legend_children,'Type'),'line'));
delete(this_legend_lines);

title('Past modality effects on current trial performance');
set(gca,'ylim',[0.5 0.75],'YTick',0:0.5:1.0);
ylabel('P(correct)','FontWeight','Normal','fontname','times');
set(gca,'XTick',[1 2 3],'XTickLabel',conditions(1:3),'FontWeight','Normal','fontname','times');
set(gca,'FontSize',16,'fontname','times');

fignum = fignum + 1;
figure(fignum); clf(fignum);
hold on;

conditions = {'low rel aud','high rel aud','vis','cc low rel aud','cc high rel aud'};
cmap = [32 255 32; 0 100 0; 0 0 0; 32 32 255; 0 0 100]/255;

%for CurrentConditionCount=1:3
%    current_condition = conditions{CurrentConditionCount};
%end

X = sort(get_errorbar_xcoords_for_grouped_bar_plot(ones(3,3)));

ref_line = plot([0.5,3.5],[0,0],'--k','linewidth',1);
set(get(get(ref_line,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

for PastConditionCount = 1:3
    past_condition = conditions{PastConditionCount};
    legend_labels{PastConditionCount} = past_condition;
    %this_past_cond_ests = coefficient_vec(39:41) -  coefficient_vec(29+PastConditionCount:3:38);
    this_past_cond_ests = coefficient_vec(29+PastConditionCount:3:38);
    this_past_cond_stderrs = model_stats.se(29+PastConditionCount:3:38);
    this_past_cond_xcoord = X(PastConditionCount:3:end);
    this_past_cond_color = cmap(PastConditionCount,:);
    
    plot(this_past_cond_xcoord,this_past_cond_ests,'o','MarkerSize',8,'MarkerFaceColor',this_past_cond_color,'MarkerEdgeColor',this_past_cond_color);
    for jj = 1:numel(this_past_cond_xcoord)
        this_plot = plot([this_past_cond_xcoord(jj), this_past_cond_xcoord(jj)],[this_past_cond_ests(jj)-1.96*this_past_cond_stderrs(jj), this_past_cond_ests(jj)+1.96*this_past_cond_stderrs(jj)],'-','LineWidth',1.5,'Color',this_past_cond_color);
        set(get(get(this_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
end

this_legend = legend(legend_labels,'fontname','times','fontsize',16,'Location','Best');
set(this_legend,'Box','off');
this_legend_children = get(this_legend,'Children');
this_legend_text = this_legend_children(strcmp(get(this_legend_children,'Type'),'text'));  %# Select the legend children of type 'text'
set(this_legend_text,{'Color'},{cmap(3,:); cmap(2,:); cmap(1,:)});    %# Set the colors
this_legend_lines = this_legend_children(strcmp(get(this_legend_children,'Type'),'line'));
delete(this_legend_lines);

title('Past modality effects on current trial performance');
%set(gca,'ylim',[0.5 0.75],'YTick',0:0.5:1.0);
ylabel('g_m_(_t_-_1_)','FontWeight','Normal','fontname','times');
set(gca,'XTick',[1 2 3],'XTickLabel',conditions(1:3),'FontWeight','Normal','fontname','times');
set(gca,'FontSize',16,'fontname','times');

% plot the strategy weights here.

success_bias_coef_ests = coefficient_vec(6:17);
success_bias_coef_stderrs = model_stats.se(6:17);

failure_bias_coef_ests = coefficient_vec(18:29);
failure_bias_coef_stderrs = model_stats.se(18:29);

coef_condition_indexes = [ones(1,4), 2*ones(1,4), 3*ones(1,4)];
coef_ss_gain_ests = [repmat(coefficient_vec(end-6),1,4), repmat(coefficient_vec(end-5),1,4), repmat(coefficient_vec(end-4),1,4)];
coef_ss_gain_stderrs = [repmat(model_stats.se(end-6),1,4), repmat(model_stats.se(end-5),1,4), repmat(model_stats.se(end-4),1,4)];

coef_abs_rate_dists = repmat(0.5:3.5,1,3);
coef_evidence_strength_ests = coef_ss_gain_ests .* coef_abs_rate_dists;
coef_evidence_strength_stderrs = coef_abs_rate_dists.*coef_ss_gain_stderrs;

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_abs_rate_dists;
x_stderrs = zeros(size(x));
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_abs_rate_dists)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_ss_gain_ests;
x_stderrs = coef_ss_gain_stderrs;
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_ss_gain_ests)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_evidence_strength_ests;
x_stderrs = coef_evidence_strength_stderrs;
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x*|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_evidence_strength_ests)]);
    
% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_abs_rate_dists;
x_stderrs = zeros(size(x));
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_abs_rate_dists)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_ss_gain_ests;
x_stderrs = coef_ss_gain_stderrs;
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_ss_gain_ests)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_evidence_strength_ests;
x_stderrs = coef_evidence_strength_stderrs;
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x*|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_evidence_strength_ests)]);


% save plots
ImageOutPath = [ImageOutDir '/' subject '_full_model_previous_modality_performance_effects.ps'];
delete(ImageOutPath);
for ImageCount = 1:fignum
    print(ImageCount,'-dpsc','-append','-r200',ImageOutPath);
end
%}

%{
% plot the strategy weights here.

success_bias_coef_ests = coefficient_vec(6:17);
success_bias_coef_stderrs = model_stats.se(6:17);

failure_bias_coef_ests = coefficient_vec(18:29);
failure_bias_coef_stderrs = model_stats.se(18:29);

coef_condition_indexes = [ones(1,4), 2*ones(1,4), 3*ones(1,4)];
coef_ss_gain_ests = [repmat(coefficient_vec(end-2),1,4), repmat(coefficient_vec(end-1),1,4), repmat(coefficient_vec(end),1,4)];
coef_ss_gain_stderrs = [repmat(model_stats.se(end-2),1,4), repmat(model_stats.se(end-1),1,4), repmat(model_stats.se(end),1,4)];

coef_abs_rate_dists = repmat(0.5:3.5,1,3);
coef_evidence_strength_ests = coef_ss_gain_ests .* coef_abs_rate_dists;
coef_evidence_strength_stderrs = coef_abs_rate_dists.*coef_ss_gain_stderrs;

% plot the strategy weights.
fignum = 1; figure(fignum); clf(fignum);
hold on;

x = coef_abs_rate_dists;
x_stderrs = zeros(size(x));
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_abs_rate_dists)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_ss_gain_ests;
x_stderrs = coef_ss_gain_stderrs;
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_ss_gain_ests)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_evidence_strength_ests;
x_stderrs = coef_evidence_strength_stderrs;
bvals = polyfit(x(:),success_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),success_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(success_bias_coef_ests)
   
    this_coef_val_est = success_bias_coef_ests(CoefCount);
    this_coef_val_stderr = success_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x*|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_s','fontsize',24,'fontname','times'); 
title('Past success strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_evidence_strength_ests)]);
    
% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_abs_rate_dists;
x_stderrs = zeros(size(x));
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_abs_rate_dists)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_ss_gain_ests;
x_stderrs = coef_ss_gain_stderrs;
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_ss_gain_ests)]);

% plot the strategy weights.
fignum = fignum+1; figure(fignum); clf(fignum);
hold on;

x = coef_evidence_strength_ests;
x_stderrs = coef_evidence_strength_stderrs;
bvals = polyfit(x(:),failure_bias_coef_ests(:),1);

plot([min(x), max(x)],[bvals(1)*min(x)+bvals(2),bvals(1)*max(x)+bvals(2)],'k--','LineWidth',1.5);
[r,p] = corr(x(:),failure_bias_coef_ests(:));
text(0.25,0.2,['r(' num2str(length(x-2)) ') = ' num2str(r) ', p = ' num2str(p)],'fontname','times','fontsize',20);

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot([this_x_est-this_x_stderr, this_x_est+this_x_stderr],[this_coef_val_est,this_coef_val_est],'-','Color',this_color,'LineWidth',1.5);
    plot([this_x_est, this_x_est],[this_coef_val_est-this_coef_val_stderr,this_coef_val_est+this_coef_val_stderr],'-','Color',this_color,'LineWidth',1.5);
    
end

for CoefCount = 1:numel(failure_bias_coef_ests)
   
    this_coef_val_est = failure_bias_coef_ests(CoefCount);
    this_coef_val_stderr = failure_bias_coef_stderrs(CoefCount);
    this_condition = coef_condition_indexes(CoefCount);
    this_color = cmap(this_condition,:);
    this_x_est = x(CoefCount);
    this_x_stderr = x_stderrs(CoefCount);
   
    plot(this_x_est,this_coef_val_est,'o','MarkerFaceColor',this_color,'MarkerEdgeColor',this_color','MarkerSize',8);
    
end
    
xlabel('g_x*|r_x - CB|','fontsize',24,'fontname','times'); ylabel('b_f','fontsize',24,'fontname','times'); 
title('Past failure strategy terms','fontsize',24,'fontname','times');
set(gca,'fontname','times','fontsize',24,'ylim',[-1.25 1.25],'xlim',[0 1.25*max(coef_evidence_strength_ests)]);

% output the plots
ImageOutPath = [ImageOutDir '/' subject '_strategy_each_modality_and_rate_dist_past_success_coefs.ps'];
delete(ImageOutPath);
for jj = 1:fignum
print(jj,'-dpsc','-append','-r200',ImageOutPath);
end
%}





%%%%%%%%%%%%%%%%%%%%%%%%%%% OLD CODE HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STRATEGY PAST TRIAL RATE-SPECIFIC EFFECTS ANALYSIS (for each modality)
%{
easy_high_rate_trials = [dat.this_click_rate] == 15 | [dat.this_click_rate] == 16;
easy_low_rate_trials = [dat.this_click_rate] == 9 | [dat.this_click_rate] == 10;

hard_low_rate_trials = [dat.this_click_rate] == 11 | [dat.this_click_rate] == 12;
hard_high_rate_trials = [dat.this_click_rate] == 13 | [dat.this_click_rate] == 14;

past_rate_type_matrix = [ones(length(dat),1), easy_low_rate_trials(:), hard_low_rate_trials(:), hard_high_rate_trials(:), easy_high_rate_trials(:)];
past_rate_type_matrix = [zeros(1,size(past_rate_type_matrix,2)); past_rate_type_matrix(1:end-1,:)];
past_rate_type_matrix(1,1) = 1;

rate_types = {'all','Easy low S','Hard low S','Hard high S','Easy high S'};
all_past_rate_type_success_history_matrix = repmat(trial_success_history(:),1,size(past_rate_type_matrix,2)) .* past_rate_type_matrix;


fignum=0;
cmap = hsv(numel(rate_types));

for ConditionCount = 1:size(trial_type_matrix,2)
    
    fignum = fignum+1; figure(fignum); clf(fignum); hold on;
    condition = conditions{ConditionCount};
    title(condition);
    
    for RateTypeCount = 1:numel(rate_types)
        
        this_color = cmap(RateTypeCount,:);
        
        if RateTypeCount == 1
            this_dat = dat( find( trial_type_matrix(:,ConditionCount ) ));
        elseif RateTypeCount == 2
            this_dat = dat( find( trial_type_matrix(:,ConditionCount) & all_past_rate_type_success_history_matrix(:,RateTypeCount) == -1 ));
        elseif RateTypeCount == 3
            this_dat = dat( find( trial_type_matrix(:,ConditionCount) & all_past_rate_type_success_history_matrix(:,RateTypeCount) == -1 ));
        elseif RateTypeCount == 4
            this_dat =  dat( find( trial_type_matrix(:,ConditionCount) & all_past_rate_type_success_history_matrix(:,RateTypeCount) == 1 ));
        elseif RateTypeCount == 5
            this_dat =  dat( find( trial_type_matrix(:,ConditionCount) & all_past_rate_type_success_history_matrix(:,RateTypeCount) == 1 ));
        end
         
    UseShiftedClickRates = 0; species = 'rat';
    [SummaryData,BootResults,diagnostics] = get_psych_curve_param(this_dat,UseShiftedClickRates,species);
    [UniqueStimVals,NumTrialsChoseHi,NumTrials] = deal(SummaryData(:,1),SummaryData(:,2),SummaryData(:,3));
    
        psych_curve_plot = plot(diagnostics.pmf(:,1), diagnostics.pmf(:,2),'-','Color',this_color,'LineWidth',3);
        
        raw_data_plot = plot(UniqueStimVals, NumTrialsChoseHi./NumTrials,'o','Color',this_color,'MarkerFaceColor',this_color,'MarkerEdgeColor',this_color);
        set(get(get(raw_data_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        FractionChoseHiStdErrs = sqrt((NumTrialsChoseHi./NumTrials).*(1-NumTrialsChoseHi./NumTrials)./NumTrials);
        
        for kk = 1:numel(UniqueStimVals)
            temp_plot = plot([UniqueStimVals(kk) UniqueStimVals(kk)],[NumTrialsChoseHi(kk)/NumTrials(kk)-FractionChoseHiStdErrs(kk) NumTrialsChoseHi(kk)/NumTrials(kk)+FractionChoseHiStdErrs(kk)],'-','Color',this_color,'LineWidth',2);
            set(get(get(temp_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        
    end

make_pretty_psych_curve(fignum,24);
set(gca,'ylim',[-1.25 1.25],'xlim',[9 16],'FontSize',24,'FontWeight','Normal','fontname','times');
this_legend = legend(rate_types,'Interpreter','none','FontSize',20,'FontWeight','Normal','Location','Best');
set(this_legend,'Box','off');
this_legend_children = get(this_legend,'Children');
this_legend_text = this_legend_children(strcmp(get(this_legend_children,'Type'),'text'));  %# Select the legend children of type 'text'
%keyboard
set(this_legend_text,{'Color'},{cmap(5,:); cmap(4,:); cmap(3,:); cmap(2,:); cmap(1,:)});    %# Set the colors
this_legend_lines = this_legend_children(strcmp(get(this_legend_children,'Type'),'line'));
delete(this_legend_lines);
    
end

% save plots
ImageOutPath = [ImageOutDir '/' subject '_strategy_past_trial_rate_specific_effects_each_modality.ps'];
delete(ImageOutPath);
for ImageCount = 1:fignum
    print(ImageCount,'-dpsc','-append','-r200',ImageOutPath);
end
%}

% STRATEGY PAST TRIAL EFFECTS AT EACH MODALITY
%{
fignum = 0; 

history_trial_types = {'all','LS','LF','RS','RF'};

% get a trial history for each different condition
all_trial_type_success_history_matrix = repmat(trial_success_history(:),1,size(trial_type_matrix,2)) .* trial_type_matrix;
all_trial_type_failure_history_matrix = repmat(trial_failure_history(:),1,size(trial_type_matrix,2)) .* trial_type_matrix;


cmap = hsv(numel(history_trial_types));

for ConditionCount = 1:size(trial_type_matrix,2)
    
    fignum = fignum+1; figure(fignum); clf(fignum); hold on;
    condition = conditions{ConditionCount};
    title(condition);
    
    
    for HistoryEventCount = 1:numel(history_trial_types)
        
        this_color = cmap(HistoryEventCount,:);
        
        if HistoryEventCount == 1
            this_dat = dat;
        elseif HistoryEventCount == 2
            this_dat = dat( find( all_trial_type_success_history_matrix(:,ConditionCount) == -1 ));
        elseif HistoryEventCount == 3
            this_dat = dat( find( all_trial_type_failure_history_matrix(:,ConditionCount) == -1 ));
        elseif HistoryEventCount == 4
            this_dat =  dat( find( all_trial_type_success_history_matrix(:,ConditionCount) == 1 ));
        elseif HistoryEventCount == 5
            this_dat =  dat( find( all_trial_type_failure_history_matrix(:,ConditionCount) == 1 ));
        end
         
    UseShiftedClickRates = 0; species = 'rat';
    [SummaryData,BootResults,diagnostics] = get_psych_curve_param(this_dat,UseShiftedClickRates,species);
    [UniqueStimVals,NumTrialsChoseHi,NumTrials] = deal(SummaryData(:,1),SummaryData(:,2),SummaryData(:,3));
    
        psych_curve_plot = plot(diagnostics.pmf(:,1), diagnostics.pmf(:,2),'-','Color',this_color,'LineWidth',3);
        
        raw_data_plot = plot(UniqueStimVals, NumTrialsChoseHi./NumTrials,'o','Color',this_color,'MarkerFaceColor',this_color,'MarkerEdgeColor',this_color);
        set(get(get(raw_data_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        FractionChoseHiStdErrs = sqrt((NumTrialsChoseHi./NumTrials).*(1-NumTrialsChoseHi./NumTrials)./NumTrials);
        
        for kk = 1:numel(UniqueStimVals)
            temp_plot = plot([UniqueStimVals(kk) UniqueStimVals(kk)],[NumTrialsChoseHi(kk)/NumTrials(kk)-FractionChoseHiStdErrs(kk) NumTrialsChoseHi(kk)/NumTrials(kk)+FractionChoseHiStdErrs(kk)],'-','Color',this_color,'LineWidth',2);
            set(get(get(temp_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        
    end

make_pretty_psych_curve(fignum,24);
set(gca,'ylim',[-1.25 1.25],'xlim',[9 16],'FontSize',24,'FontWeight','Normal','fontname','times');
this_legend = legend(history_trial_types,'Interpreter','none','FontSize',20,'FontWeight','Normal','Location','Best');
set(this_legend,'Box','off');
this_legend_children = get(this_legend,'Children');
this_legend_text = this_legend_children(strcmp(get(this_legend_children,'Type'),'text'));  %# Select the legend children of type 'text'
%keyboard
set(this_legend_text,{'Color'},{cmap(5,:); cmap(4,:); cmap(3,:); cmap(2,:); cmap(1,:)});    %# Set the colors
this_legend_lines = this_legend_children(strcmp(get(this_legend_children,'Type'),'line'));
delete(this_legend_lines);
    
end

% save plots
ImageOutPath = [ImageOutDir '/' subject '_past_trial_strategy_effects_at_each_modality.ps'];
delete(ImageOutPath);
for ImageCount = 1:fignum
    print(ImageCount,'-dpsc','-append','-r200',ImageOutPath);
end
%}

% STRATEGY -- MODALITY OF PAST TRIAL EFFECTS
%{
fignum = 0; 

history_trial_types = {'all','LS','LF','RS','RF'};

past_trial_type_matrix = [zeros(1,size(trial_type_matrix,2)); trial_type_matrix(1:end-1,:)];

% get a trial history for each different condition
all_trial_type_success_history_matrix = repmat(trial_success_history(:),1,size(trial_type_matrix,2)) .* past_trial_type_matrix;
all_trial_type_failure_history_matrix = repmat(trial_failure_history(:),1,size(trial_type_matrix,2)) .* past_trial_type_matrix;


cmap = hsv(numel(history_trial_types));

for ConditionCount = 1:size(past_trial_type_matrix,2)
    
    fignum = fignum+1; figure(fignum); clf(fignum); hold on;
    condition = conditions{ConditionCount};
    title(condition);
    
    
    for HistoryEventCount = 1:numel(history_trial_types)
        
        this_color = cmap(HistoryEventCount,:);
        
        if HistoryEventCount == 1
            this_dat = dat;
        elseif HistoryEventCount == 2
            this_dat = dat( find( all_trial_type_success_history_matrix(:,ConditionCount) == -1 ));
        elseif HistoryEventCount == 3
            this_dat = dat( find( all_trial_type_failure_history_matrix(:,ConditionCount) == -1 ));
        elseif HistoryEventCount == 4
            this_dat =  dat( find( all_trial_type_success_history_matrix(:,ConditionCount) == 1 ));
        elseif HistoryEventCount == 5
            this_dat =  dat( find( all_trial_type_failure_history_matrix(:,ConditionCount) == 1 ));
        end
         
    UseShiftedClickRates = 0; species = 'rat';
    [SummaryData,BootResults,diagnostics] = get_psych_curve_param(this_dat,UseShiftedClickRates,species);
    [UniqueStimVals,NumTrialsChoseHi,NumTrials] = deal(SummaryData(:,1),SummaryData(:,2),SummaryData(:,3));
    
        psych_curve_plot = plot(diagnostics.pmf(:,1), diagnostics.pmf(:,2),'-','Color',this_color,'LineWidth',3);
        
        raw_data_plot = plot(UniqueStimVals, NumTrialsChoseHi./NumTrials,'o','Color',this_color,'MarkerFaceColor',this_color,'MarkerEdgeColor',this_color);
        set(get(get(raw_data_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        FractionChoseHiStdErrs = sqrt((NumTrialsChoseHi./NumTrials).*(1-NumTrialsChoseHi./NumTrials)./NumTrials);
        
        for kk = 1:numel(UniqueStimVals)
            temp_plot = plot([UniqueStimVals(kk) UniqueStimVals(kk)],[NumTrialsChoseHi(kk)/NumTrials(kk)-FractionChoseHiStdErrs(kk) NumTrialsChoseHi(kk)/NumTrials(kk)+FractionChoseHiStdErrs(kk)],'-','Color',this_color,'LineWidth',2);
            set(get(get(temp_plot,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        
    end

make_pretty_psych_curve(fignum,24);
set(gca,'ylim',[-1.25 1.25],'xlim',[9 16],'FontSize',24,'FontWeight','Normal','fontname','times');
this_legend = legend(history_trial_types,'Interpreter','none','FontSize',20,'FontWeight','Normal','Location','Best');
set(this_legend,'Box','off');
this_legend_children = get(this_legend,'Children');
this_legend_text = this_legend_children(strcmp(get(this_legend_children,'Type'),'text'));  %# Select the legend children of type 'text'
%keyboard
set(this_legend_text,{'Color'},{cmap(5,:); cmap(4,:); cmap(3,:); cmap(2,:); cmap(1,:)});    %# Set the colors
this_legend_lines = this_legend_children(strcmp(get(this_legend_children,'Type'),'line'));
delete(this_legend_lines);
    
end

% save plots
ImageOutPath = [ImageOutDir '/' subject '_previous_modality_strategy_effects.ps'];
delete(ImageOutPath);
for ImageCount = 1:fignum
    print(ImageCount,'-dpsc','-append','-r200',ImageOutPath);
end
%}

end %EOF

function [pse_est,pse_stderr] = get_pse_est_and_stderr(mcestimates,rates)
% Assumes model is cumulative Gaussian with guess/lapse rates and
% 'ab' core.
    
    for jj=1:size(mcestimates,1)
        theta = mcestimates(jj,:);
        this_pmf = theta(4) + (1-theta(3)-theta(4)).*0.5*(1+erf(1./sqrt(2).*(rates-theta(1))./theta(2)));
        [junk,pse_index] = min(abs(this_pmf - 0.5));
        pse_ests(jj) = rates(pse_index);
    end
    
    pse_est = mean(pse_ests);
    pse_stderr = std(pse_ests);

end % EOF
