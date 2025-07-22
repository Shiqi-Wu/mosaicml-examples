export INPUT_DIR="/mnt/data/dnabert_2_pretrain/dev.txt"
export OUTPUT_DIR="/mnt/data/dnabert_2_pretrain/val"
export TOKENIZER_NAME="zhihan1996/DNABERT-2-117M"
export MAX_LENGTH=128

python src/convert_dataset_dnabert2.py \
    --input_txt $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --tokenizer $TOKENIZER_NAME \
    --max_length $MAX_LENGTH

# python src/text_data.py --local_path /mnt/data/dnabert_2_pretrain --tokenizer zhihan1996/DNABERT-2-117M
