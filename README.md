# SLDS for human resting state fMRI

## Folder Descriptions
#### HCP_download_and_preprocessing     
-scripts to download and preprocess resting state fMRI data from Human Connectome Project (https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)

#### model_selection     
-script to fit cross-validation models  
-scripts to choose number of states and number of latent dimensions

#### Final_model          
-multi-level SLDS model fitted from 500 HCP subjects, used for main results reporting

#### analysis_scripts  
-scripts to run analysis, summarize from data and the fitted SLDS model, conduct statistical tests, generate figures

#### optimization  
-additional script to fit multi-level slds for fMRI data, based on the original 'ssm' package by Linderman (please also refer to https://github.com/lindermanlab/ssm)

More detailed information can be found in the 'Readme.txt' within individual folder.

## Acknowledgments
This project makes use of the following open-source packages:

The original ssm package developed by Linderman:
https://github.com/lindermanlab/ssm   
https://proceedings.mlr.press/v119/zoltowski20a.html

We use a composite 254-ROI parcellation (cortex+subcortex) from the Melbourne atlas: 
https://github.com/yetianmed/subcortex/tree/master/Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric
https://www.nature.com/articles/s41593-020-00711-6

The cortex part uses 200-ROI Schaefer-parcellation and Yeo-17Network grouping:
https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
https://academic.oup.com/cercor/article/28/9/3095/3978804

