data_path=$(realpath $1)

. bash/run_openpose.sh $data_path/image $data_path/openpose
. bash/run_densepose.sh $data_path/image $data_path/image-densepose
. bash/run_human_parsing.sh $data_path/image $data_path/image-parse-v3

conda run -n StableVITON python /home/jun/StableVITON/custom_data_preprocessing/script/agnostic_map.py \
    $data_path $data_path/agnostic-v3.2 $data_path/agnostic-mask