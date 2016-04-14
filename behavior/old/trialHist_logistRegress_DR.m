% FN adapted from C:\Users\fnajafi\Documents\repoland\playgrounds\anne\optical\opticalStimLogisticRegress.m
%
% David Raposo
% Updated Sep 16, 2015
%
% Z = B0 + G * (rate(t) - boundary) + Bf * failure(t-1) + Bs * success(t-1)
%
% OUTPUT: B = [B0_aud, B0_mult, B0_vis, G_aud, G_mult, G_vis, Bs, Bf]
%
% "separate_by_madality" can be -1, 0 or 1.
% 1 will calculate Bs and Bf for the 3 different modalities, based on the
% modality of the current trials. -1 will calculate the same parameters for the 3
% modalities also, but based on the modality of the previous trial. 0 will
% calcule only one Bs and one Bf, irrespective of modality.
%
function [B, deviance, stats, X, Y, pvals] = trialHist_logistRegress_DR (data, separate_by_modality)

if nargin < 2
	separate_by_modality = 0;
end

boundary = data(1).categoryBoundaryHz; %12.5; % We need to have the boundary saved in the behavioral data!

bzero = zeros(length(data),3*2);
rateDiff = zeros(length(data),3*2);
success = zeros(length(data),3*2);
failure = zeros(length(data),3*2);
response = zeros(length(data),1);

% Need to make sure the first trial is not taken into account
% in the regression.
response(1) = nan;

n_ignore_trials = 0;

if length(data) < 4
    error('there is no data')
end;

[nevents, stimdur, stimrate, stimtype] = setStimRateType(data);

for t = 2:length(data)
   
    modality = data(t).visualOrAuditory + 2;
    is_optical = 0; % data(t).isOpticalTrial;
    cond = modality + 3 * is_optical;
    high_rate_port = 1;
    if strcmp(data(t).highRateChoicePort,'left') | strcmp(data(t).highRateChoicePort,'L')
        high_rate_port = -1;       
    end
    
    bzero(t,cond) = 1;
    rateDiff(t,cond) = stimrate(t)-boundary;
%     if modality == 1
%         rateDiff(t,cond) = data(t).nAuditoryEvents - boundary;
%     else
%         rateDiff(t,cond) = data(t).nVisualEvents - boundary;
%     end
    
    response(t) = data(t).responseSideIndex - 1; % This keeps the NaN's, which is what I want.
    if high_rate_port == -1 & ~isnan(response(t))
    
        rateDiff(t,cond) = -1* rateDiff(t,cond); %added by AKC so that sensitivity means the same thing for the flipped contingency rats. 
        
        %Commented out by AKC on 10/01/2015      
        % response(t) = ~response(t); %So responses get flipped if this is a flipped-contingency rat
        %so normally 1 is left and 2 is right. he subtracts 1 to make 0 for
        %left and 1 for right. So for normal animals, a B_0<1 means that
        %the animals has a leftwards, low-rate bias. 
        
        %Then if this is the flipped contingency, 0
        %is right at 1 is left. So this means that for these animals, B_0<1
        %means that the animals have a rightwards bias because 0 is right.
        %a Rightwards bias is a contra 
        
        %This changing of sign means that the data are in terms of the
        %RATE. So Beta > 0 always means a high rate bias. You might want
        %this, but you also might want the data in terms of SIDE; you might
        %want to know if the animal has not a rate bias but a SIDE bias in
        %which case these should not be flipped! 
        
  
    end
    
    
    
   
    if separate_by_modality == 0
        cond = 1;
    % Not sure if the following is correct
    elseif separate_by_modality == -1 % Separate by modality of previous trial.
        modality = data(t-1).visualOrAuditory + 2;
        cond = modality + modality * is_optical - 1
    end

%     if data(t-1).earlyWithdrawal | data(t-1).didNotChoose
        failure(t,cond) = 0;
        success(t,cond) = 0;
        
    if data(t-1).outcome==1
%         failure(t,cond) = 0;
        if data(t-1).responseSideIndex == 1
            success(t,cond) = -1 * high_rate_port;  %AKC added the "* high_rate_port" so that these numbers would be flipped for reverse contingency animals (like js15).
        elseif data(t-1).responseSideIndex == 2
            success(t,cond) = 1 * high_rate_port;
        end
        
    elseif data(t-1).outcome==0 % There should be only one case remaining: rat chose the wrong port (punish).
%         success(t,cond) = 0;
        if data(t-1).responseSideIndex == 1
            failure(t,cond) = -1 * high_rate_port;
        elseif data(t-1).responseSideIndex == 2
            failure(t,cond) = 1 * high_rate_port;
        end
    end
        
 
    % ignore first trial of every session that was lumped together
    % you can use this to ignore other specific trials for any reason
    if data(t).trialId == 1 % | (ismember(data(t).nAuditoryEvents,[10 11]) & data(t).correctSide == 2)
        response(t) = nan;
        n_ignore_trials = n_ignore_trials + 1;
    end
    
end

n_ignore_trials;


success = success .* high_rate_port;
failure = failure .* high_rate_port;

if separate_by_modality == 0
    success = success(:,1);
    failure = failure(:,1);
end


% rateDiff = rateDiff ./ max(max(rateDiff));
% X = [bzero rateDiff success failure];
X = [bzero(:,2) rateDiff(:,2) success failure];
% X = [bzero rateDiff];
Y = response;
[B, deviance, stats] = glmfit(X, Y, 'binomial', 'constant', 'off');

%Not sure what to do here; I only have the SE from glmfit; it is computed by diag(sqrt(stats.covb)). So I assumed
%(possible incorrectly) that they just divided the std by sqrt of the
%number of trials. But I am troubled by this because the degrees of freedom
%in glmfit is a lot less than the number of trials. Anyone, once I do that,
%it is the standard deviation, and the I just use sqrt(a^2 + b^2)
%becausethe variances add. So then that is the standard deviation on the
%difference and I use that for the t-statistic. 

%This will be for vision only;
if ~isempty(B(3)) %Make sure we had visual trials for that day. 
numTrialsSqrt = sqrt(size(X,1));
%First do visual trials
ErrorOnStimControlDifference = sqrt((numTrialsSqrt*stats.se(3:6:24)).^2 + (numTrialsSqrt*stats.se(6:6:24)).^2)./numTrialsSqrt;
%This is for a 1-tailed test! Not at all right for b_sucess and b_fail for
%which we have no hypothesis. 
[pvals_vis stat] = make_own_p_value_from_tstatistic(B(3:6:24) - B(6:6:24),ErrorOnStimControlDifference ,size(X,1));
pvals(3).modality = 'visual';
pvals(3).pval = pvals_vis;
pvals(3).betas = B(6:6:24) - B(3:6:24);
pvals(3).numtrials = size(find([data.visualOrAuditory] == 1),2);
end

%This is for audition 
if ~isempty(B(1)) %Make sure we had auditory trials for that day. 
%Then do auditory trials
ErrorOnStimControlDifferenceAud = sqrt((numTrialsSqrt*stats.se(1:6:24)).^2 + (numTrialsSqrt*stats.se(4:6:24)).^2)./numTrialsSqrt;
[pvals_aud stat] = make_own_p_value_from_tstatistic(B(1:6:24) - B(4:6:24),ErrorOnStimControlDifference ,size(X,1));
pvals(1).modality = 'auditory';
pvals(1).pval = pvals_aud;
pvals(1).betas = B(4:6:24) - B(1:6:24);
pvals(1).numtrials = size(find([data.visualOrAuditory] == -1),2); %Auditory really is -1. Lame. 
end;


%You could plot this if you wanted! remember that the Bs are the parameters
%but by the model; I should be able to just do this using feval, but
%whatever. 

%fit = B(1) + (B(7).*xax)
%xax = [-3.5:0.1:3.5]
%choice_model_response_predictions = 1./(1+exp(-1*fit));
%plot(xax,choice_model_response_predictions)


