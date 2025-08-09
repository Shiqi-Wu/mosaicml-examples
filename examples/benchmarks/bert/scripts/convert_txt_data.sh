export INPUT_DIR="/mnt/data/dnabert_2_pretrain/dev.txt"
export OUTPUT_DIR="/mnt/data/dnabert_2_pretrain_full/val"
export TOKENIZER_CKPT="./saved_models/tokenizer_ckpt"
export MAX_LENGTH=128

python src/convert_dataset_dnabert2.py \
    --input_txt $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --tokenizer $TOKENIZER_CKPT \
    --max_length $MAX_LENGTH

# python src/text_data.py --local_path /mnt/data/dnabert_2_pretrain --tokenizer zhihan1996/DNABERT-2-117M
