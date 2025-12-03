proj_path=/home/xzhou126/HCP_dataset
subj_ids_file=${proj_path}/scripts/hcp_subjects_with_all_rsfMRI.txt
atlas=${proj_path}/masks/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S4_MNI152NLin6Asym_2mm.nii.gz
out=${proj_path}/resting_state_denoised_cortex-subcortex_254_rois_awscli

mkdir -p ${out}

declare -a runs=("LR" "RL")

for pid in `cat ${subj_ids_file}`
do

	rm -rf ${proj_path}/scripts/tmp
	mkdir ${proj_path}/scripts/tmp
	
	for ses in `seq 1 1 2`
	do	
		
		for run in "${runs[@]}"
		do	
		
			# download the file
			aws s3 cp s3://hcp-openaccess/HCP_1200/$pid/MNINonLinear/Results/rfMRI_REST${ses}_${run}/rfMRI_REST${ses}_${run}_hp2000_clean.nii.gz ${proj_path}/scripts/tmp
			
			# extract ROI timeseries from the file
			3dROIstats -quiet -mask $atlas ${proj_path}/scripts/tmp/rfMRI_REST${ses}_${run}_hp2000_clean.nii.gz > ${out}/${pid}_rfMRI_REST${ses}_${run}.1D										
		
		done
	done
	
done 
