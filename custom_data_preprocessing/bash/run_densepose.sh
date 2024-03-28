detectron_path="/home/jun/StableVITON/custom_data_preprocessing/detectron2"

conda run -n densepose python $detectron_path/projects/DensePose/apply_net.py \
    show \
    $detectron_path/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    $1 dp_segm \
    --output $2