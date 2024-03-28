schp_path="/home/jun/StableVITON/custom_data_preprocessing/Self-Correction-Human-Parsing"

conda run -n schp python $schp_path/simple_extractor.py \
    --dataset lip \
    --model-restore $schp_path/pretrain_model/exp-schp-201908261155-lip.pth \
    --input-dir $1 \
    --output-dir $2