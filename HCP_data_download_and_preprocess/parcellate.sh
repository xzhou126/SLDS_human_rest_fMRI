proj_path=/home/xzhou126/HCP_dataset
download_path=/media/xzhou126/HCP_resting_state_denoised
atlas=${proj_path}/masks/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz
out=${proj_path}/rfMRI_denoised_cortex_only_200_17

temp_folder=${proj_path}/tmp
rm -r ${temp_folder}

zip_files=${download_path}/??????_3T_rfMRI_REST?_fixextended.zip

for fpath in `ls ${zip_files}`
do
	echo "#########################################################"
	echo processing ${fpath}
	echo "#########################################################"
	echo ""
	echo ""
	
	mkdir -p ${temp_folder}
	
	fname=`basename $fpath`
	pid="${fname:0:6}"
	
	case "$fname" in
	*REST1*)
		sid=1
	;;
	*REST2*)
		sid=2
	;;
	esac

	file1=$pid/MNINonLinear/Results/rfMRI_REST${sid}_LR/rfMRI_REST${sid}_LR_hp2000_clean.nii.gz
	file2=$pid/MNINonLinear/Results/rfMRI_REST${sid}_RL/rfMRI_REST${sid}_RL_hp2000_clean.nii.gz
	
	unzip -j ${fpath} ${file1} ${file2} -d ${temp_folder}
	
	# obtaining average timeseries for each ROI in the atlas
	3dROIstats -quiet -mask $atlas ${temp_folder}/rfMRI_REST${sid}_LR_hp2000_clean.nii.gz > ${out}/${pid}_rfMRI_REST${sid}_LR.1D
	3dROIstats -quiet -mask $atlas ${temp_folder}/rfMRI_REST${sid}_RL_hp2000_clean.nii.gz > ${out}/${pid}_rfMRI_REST${sid}_RL.1D
	
	rm -r ${temp_folder}	
	
	echo "======================= D O N E ======================="
	echo ""
	echo ""
done
