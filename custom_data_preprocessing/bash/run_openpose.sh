openpose_path="/home/public/openpose"

image_path="/home/jun/StableVITON/custom_data_preprocessing/data/image"
output_path_prefix="/home/jun/StableVITON/custom_data_preprocessing/data/openpose"

cd $openpose_path
./build/examples/openpose/openpose.bin \
    --image_dir $image_path \
    --hand \
    --disable_blending \
    --display 0 \
    --write_json $output_path_prefix"_json" \
    --write_images $output_path_prefix"_img" \
    --num_gpu 1 --num_gpu_start 0
cd - > /dev/null