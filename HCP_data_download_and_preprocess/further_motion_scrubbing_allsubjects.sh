proj_path=/home/xzhou126/HCP_dataset
subj_ids_file=${proj_path}/scripts/hcp_subjects_with_all_rsfMRI.txt


declare -a runs=("LR" "RL")

for pid in `cat ${subj_ids_file}`
do
	for ses in `seq 1 1 2`
	do	
		for run in "${runs[@]}"
		do	
			csh further_motion_scrubbing.sh ${pid} REST${ses}_${run}
		done
	done
done	
