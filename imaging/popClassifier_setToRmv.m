% Set toRmv, ie trials in which go tone happened before ep_ms(end) and trials
% that have <th_stim_dur duration. 



% clean up timeStimOnset for setting stim-aligned traces that will be used for SVM classifying of current choice (make sure go tone is not within ep and stimDur is long enough (800ms))
% Set ep (SVM will be trained on ep frames to decode choice) and set to nan timeStimOnset of those trials 
% that have issues with this ep: ie their go tone is within ep, or their stimDur is not long enough (800ms).

% include in timeStimOnset trials that are:
% stimulus duration >= th_stim_dur 
% and go tone after ep_ms(2)

% th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset
% ep_ms = [700 900]; % [500 700]; % rel2 stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during [500 700]ms after stimulus onset. Why?
% bc we think at this window choice signals might already exist, also we
% can see if they continue during [700 stimOffset], ie mostly [700 1000]ms.
% (ie if decoder generalizes to time points beyond ep)
% you can also try [600 800].... but for now lets go with [500 700].

%%
% now make sure in no trial go tone happened before the end of ep:
i = timeCommitCL_CR_Gotone <= ep_ms(end);
if sum(i)>0
    fprintf('Excluding %i trials from timeStimOnset bc their goTone is earlier than ep end\n', sum(i))
%     timeStimOnset(i) = NaN;  % by setting to nan, the aligned-traces of these trials will be computed as nan.
else
    fprintf('No trials with go tone before the end of ep. Good :)\n')
end

% now make sure trials that you use for SVM (decoding upcoming choice from
% neural responses during stimulus) have a certain stimulus duration. Of
% course stimulus needs to at least continue until the end of ep. 
% go with either 900 or 800ms. Since the preference is to have at least
% ~100ms after ep which contains stimulus and without any go tones, go with 800ms
% bc in many sessions go tone happened early... so you will loose lots of
% trials if you go with 900ms.

% th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset

figure; hold on
plot(timeCommitCL_CR_Gotone - timeStimOnset)
plot(timeStimOffset - timeStimOnset)
plot([1 length(timeCommitCL_CR_Gotone)],[th_stim_dur th_stim_dur],'g')
ylabel('Time relative to stim onset (ms)')
legend('stimOffset','goTone', 'th\_stim\_dur')
minStimDurNoGoTone = min(timeCommitCL_CR_Gotone - timeStimOnset); % this is the duration after stim onset during which no go tone occurred for any of the trials.
cprintf('blue', 'minStimDurNoGoTone = %.2f ms\n', minStimDurNoGoTone)


% exclude trials whose stim duration was < th_stim_dur
j = (timeStimOffset - timeStimOnset) < th_stim_dur;
if sum(j)>0
    fprintf('Excluding %i trials from timeStimOnset bc their stimDur-without-goTone < 800ms\n', sum(j))
%     timeStimOnset(j) = NaN;
else
    fprintf('No trials with stimDur-w/out-goTone < 800ms. Good :)\n')
end


% fprintf('#isnan(timeStimOnset0)= %i;  #isnan(timeStimOnset)= %i\n', sum(isnan(timeStimOnset)), sum(isnan(timeStimOnset)))

% how about trials that go tone happened earlier than stimulus offset?
% I don't think they cause problems for SVM training. Except that when you
% look at projections (neural responses projected onto SVM weights) you
% need to have in mind that go tone may exist before stim ends.
% so we don't exclude them but lets just check what is the duration after
% stim onset in which no go tone occurred for any of the trials.
toRmv = (i+j)~=0;
% final_minStimDurNoGoTone = min(timeCommitCL_CR_Gotone(~toRmv) - timeStimOnset(~toRmv)); % for trials that you are including in SVM, this is the duration after stim onset during which no go tone occurred for any of these trials.
% fprintf('final_minStimDurNoGoTone = %.2f ms\n', final_minStimDurNoGoTone)
