conda run -n schp python Self-Correction-Human-Parsing/simple_extractor.py \
    --dataset lip \
    --model-restore Self-Correction-Human-Parsing/pretrain_model/exp-schp-201908261155-lip.pth \
    --input-dir $1 \
    --output-dir $2