conda run -n schp python Self-Correction-Human-Parsing/simple_extractor.py \
    --dataset lip \
    --model-restore Self-Correction-Human-Parsing/pretrain_model/exp-schp-201908261155-lip.pth \
    --input-dir data/image \
    --output-dir data/image-parse-v3