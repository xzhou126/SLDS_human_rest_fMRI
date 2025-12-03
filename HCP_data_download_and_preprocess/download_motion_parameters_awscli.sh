proj_path=/home/xzhou126/HCP_dataset
subj_ids_file=${proj_path}/scripts/hcp_subjects_with_all_rsfMRI.txt
out=${proj_path}/resting_state_denoised_cortex-subcortex_254_rois_awscli/movement_regressors/

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
			aws s3 cp s3://hcp-openaccess/HCP_1200/$pid/MNINonLinear/Results/rfMRI_REST${ses}_${run}/Movement_Regressors.txt ${proj_path}/scripts/tmp
			aws s3 cp s3://hcp-openaccess/HCP_1200/$pid/MNINonLinear/Results/rfMRI_REST${ses}_${run}/Movement_Regressors_dt.txt ${proj_path}/scripts/tmp
			
			# move downloaded files
			mv ${proj_path}/scripts/tmp/Movement_Regressors.txt ${out}/${pid}_rfMRI_REST${ses}_${run}_motion.txt
			mv ${proj_path}/scripts/tmp/Movement_Regressors_dt.txt ${out}/${pid}_rfMRI_REST${ses}_${run}_motion_dt.txt
		
		done
	done
	
done 
