export INPUT_DIR="/mnt/data/dnabert2_pretrain_raw/train.txt"
export OUTPUT_DIR="/mnt/data/dnabert2_pretrain_lmdb/train"
export TOKENIZER_CKPT="./saved_models/tokenizer_ckpt"
export MAX_LENGTH=128

python src/convert_dataset_dnabert2_lmdb.py \
    --input_txt $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --tokenizer $TOKENIZER_CKPT \
    --max_length $MAX_LENGTH \
    --max_length 128 \
    --map_size_gb 256 \
    --commit_interval 50000 \
    --emit_index_mapping /mnt/data/dnabert2_pretrain_lmdb/train/present_indices.npy
