data_path=$(realpath $1)
sh_path=$(dirname $0)
py_path=$sh_path/../script

$sh_path/run_openpose.sh $data_path/image $data_path/openpose
$sh_path/run_densepose.sh $data_path/image $data_path/image-densepose
$sh_path/run_human_parsing.sh $data_path/image $data_path/image-parse-v3


conda run -n StableVITON python $py_path/agnostic_map.py \
    $data_path $data_path/agnostic-v3.2 $data_path/agnostic-mask