# imaging_decisionMaking_exc_inh
The scripts correspond to my postdoc project "Excitatory and Inhibitory Subnetworks Are Equally Selective during Decision-Making and Emerge Simultaneously during Learning", published in Neuron, 2019: https://www.ncbi.nlm.nih.gov/pubmed/31753580" 
(PDF file of the paper is available here: https://www.researchgate.net/profile/Farzaneh_Najafi4/research).

The data are available at CSHL repository: http://repository.cshl.edu/36980/


## Codes to generate the Figures in the paper
Below, you can find a description of the codes to generate the Figures in the paper.  
(Note: figure numbers may not match those in the final published paper; however, a description for each figure is provided below, which can help to identify them).

### Fig 1 
### PMF:  
farznaj/imaging_decisionMaking_exc_inh/behavior/PMF_allmice.m

### FOV images:
fni18, 151217 (scale: 50 um: (50*512)/580])

### PSTHs:  
farznaj/imaging_decisionMaking_exc_inh/imaging/avetrialAlign_plotAve_trGroup.m

### Inferred spikes:
farznaj/imaging_decisionMaking_exc_inh/utils/lassoClassifier/excInh_Frs.py

### Heatmap of sorted neural activity:
fni17, 151015
avetrialAlign_plotAve_trGroup (in imaging_postproc.m, set mouse name, and run the imaging_prep_analysis section, stopping at avetrialAlign_plotAve_trGroup, line 1056) 

### FRs:
farznaj/imaging_decisionMaking_exc_inh/utils/laassoClassifier/excInh_FRs.py


### Fig 2
### ROC:
Set vars in:
farznaj/imaging_decisionMaking_exc_inh/imaging/choicePref_ROC_exc_inh_plots_setVars.m

Example session AUC histogram: fni16, 151029
code gets called in line 340 of: choicePref_ROC_exc_inh_plotsEachMouse.m
outcome2ana = 'corr'; %''; % 'corr'; 'incorr'; '';
doChoicePref = 0; %2; 


### Fraction choice tuned:
choicePref_ROC_exc_inh_plotsAllMice.m
section: Fractions of significantly choice-tuned neurons
~line 340.


Example mouse absDevAUC time course: fni16   (doChoicePref = 2;

Example day AUC of corr, incorr: fni16; '151029_1-2' (last day, day 45) (doChoicePref = 0;)
run choicePref_ROC_exc_inh_plots_setVars.m once with outcome2ana = corr, another time with incorr. Then run the first section of choicePref_ROC_exc_inh_plotsEachMouse
(which calls choicePref_ROC_exc_inh_plotsEachMouse_corrIncorr)



### ROC controlling for FR values of exc and inh:
run script:
choicePref_ROC_exc_inh_plotsAllMice_sameFR


### Fraction choice selective neurons; time course; 
fni16
the following section 
Plot fraction choice-selective neurons averaged across days
in code choicePref_ROC_exc_inh_plotsEachMouse



### Fig 3
svm_excInh_trainDecoder_eachFrame_plots.py

panel B (example class accuracies): Figure: curr_chAl_day151015_exShfl3_171010-112112_sup.pdf
fni17, 1 session (151015); average and st error across cross validation samples; for exc, only one example excitatory sample is used (exShfl3) so the error bar matches that of inh and allN.

Event time distributions: eventTimesDist.py

Weights: svm_excInh_trainDecoder_eachFrame_plotWeights.py


### Supp Figure:
### Corr, incorr (SVM trained on correct trials, tested on incorrect trials):
svm_excInh_trainDecoder_eachFrame_testIncorrTrs_plots.py
Stimulus category decode:
svm_excInh_trainDecoder_eachFrame_plots.py


### Fig 4 (stability):
svm_excInh_trainDecoder_eachFrame_stabTestTimes_plots.py


### Fig 5
Panel A: example session: fni16, 151029
PW correlations:  corr_excInh_plots.m ; read comments on the top of the script for the scripts you need to run beforehand.

svm_excInh_trainDecoder_eachFrame_plots
addNs_ROC = 1
set shflTrsEachNeuron first to 1, and run the code; then to 0, and run the code.

Use the following script for the plots of all mice (change in CA after breaking noise correlations):
svm_excInh_trainDecoder_eachFrame_addNs1by1ROC_sumAllMice_plots


### Fig 6
svm_excInh_trainDecoder_eachFrame_plots


### Supp Figs:
### Running and licking: tracesAlign_wheelRev_lick_classAccur_plots.m
Fraction choice selective for early and late days: choicePref_ROC_exc_inh_plotsAllMice.m
(at the end of the script)

Example sessions for classification accuracy (time course):
svm_excInh_trainDecoder_eachFrame_plots.py
fni17, sessions: 151014 – 151029 – 151022 – 151008 – 151020 - 150903
(not used, but pretty good: 151026 – 151021 – 151013 – 150918)


------
### Temporal epoch tuning:
temporalEpochTuning_allSess.m



