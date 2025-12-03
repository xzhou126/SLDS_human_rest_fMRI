Resting state fMRI data were obtained from the Human Connectome Project S1200 release 
(https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)
that utilized ICA+FIX preprocessing pipeline
(https://www.sciencedirect.com/science/article/pii/S1053811913005338)

Scripts for downloading and further preprocessing are available in 'HCP_data_download_and_preprocess'.

Processed ROI timeseries were stored in this folder (see 'preprocess_roi_timeseries.ipynb'), total size about 10GB;  
subject IDs are saved as 'tags' to be passed to fit multi-level SLDS as well as later analysis:
1.roi_timeseries_rsfMRI_HCP_model_selection 
2.tags_rsfMRI_HCP_model_selection
3.roi_timeseries_rsfMRI_HCP_held_out
4.tags_rsfMRI_HCP_held_out

