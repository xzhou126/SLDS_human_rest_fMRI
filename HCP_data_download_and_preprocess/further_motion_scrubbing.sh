#!/bin/csh
############################## Initialize parameters ##############################
set subj_name = "$1"
set run_name = "$2"

set proj_path = /home/xzhou126/HCP_dataset
set out = ${proj_path}/resting_state_denoised_cortex-subcortex_254_rois_awscli/motion_scrubbed_timeseries
mkdir -p $out

echo "================================================"
echo "--------- Processing Subject $subj_name ----------"
echo "================================================"

set inputpath = "$proj_path"/resting_state_denoised_cortex-subcortex_254_rois_awscli
set input = "$inputpath"/"$subj_name"_rfMRI_"$run_name".1D
set fileRawMotion = "$inputpath"/movement_regressors/"$subj_name"_rfMRI_"$run_name"_motion.txt
# set fileDerMotion = "$inputpath"/movement_regressors/"$subj_name"_rfMRI_"$run_name"_motion_dt.txt
#set fileCensMotion = "$ICApath"/"$subj_name"_MotionCensor_1.1mm"$inputFile_suffix".txt

# multimask to use for extracting roi timeseries from voxelwise nifti files
#set goodROIMask = Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz 
#rm -rf "$out"/"$subj_name"_meanTS.1D
#3dROIstats -quiet -mask $goodROIMask $input > $out/"$subj_name"_meanTS.1D

setenv AFNI_1D_TIME YES
setenv AFNI_1D_TIME_TR 0.72


echo "========================================================="
echo "        Analysis of $subj_name from file: $input"
echo "========================================================="

# set event_base = "CSPLIN(-7.5, 11.25, 16)"
# set encounter_base = "CSPLIN(-7.5, 5, 11)"

# rm -rf "$out"/"$subj_name"_meanTS.1D
# 3dROIstats -quiet -mask $goodROIMask $input > $out/"$subj_name"_meanTS.1D

############################## Doing first stage regression (censor shock, motion, drifts) ##############################
3dDeconvolve -overwrite \
        -input "$input"\' \
        -noFDR \
        # -polort A \
        -local_times \
#       -censor "$fileCensMotion" \
#       -concat "$runConcatInfo" \
        -ortvec "$fileRawMotion" rawMotion \
#       -ortvec "$fileDerMotion" derMotion \
        -x1D "$out"/"$subj_name"_Main_block_deconv.x1D \
#       -x1D_uncensored "$out"/"$subj_name"_Main_block_deconv_uncensored.x1D \
#       -errts "$out"/"$subj_name"_resids.nii.gz \
#       -bucket "$out"/"$subj_name"_bucket.nii.gz \
#       -cbucket "$out"/"$subj_name"_betas.nii.gz \
        -x1D_stop

# echo "***** Running 3dREMLfit *****"
3dREMLfit -matrix "$out"/"$subj_name"_Main_block_deconv.x1D \
        -input "$input"\' \
    	-overwrite \
    	-noFDR \
#    -Rbeta "$out"/"$subj_name"_betas_REML.1D \
#    -Rbuck "$out"/"$subj_name"_bucket_REML.1D \
#    -Rvar "$out"/"$subj_name"_bucket_REMLvar.1D \
#	-Rerrts "$out"/"$subj_name"_resids_REML.1D \
	-Rerrts "$out"/"$subj_name"_"$run_name"_resids_REML.1D
#    -Rwherr "$out"/"$subj_name"__wherrs_REML.1D \

rm "$out"/"$subj_name"_meanTS.1D
rm "$out"/"$subj_name"_Main_block_deconv.x1D

echo "Exiting..."
exit
