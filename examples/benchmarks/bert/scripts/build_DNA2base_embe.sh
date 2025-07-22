export SAVE_PATH='./saved_models'
export TOKENIZER_NAME="zhihan1996/DNABERT-2-117M"
mkdir -p $SAVE_PATH

python src/dna_text_biembedder.py \
    --tokenizer_name $TOKENIZER_NAME \
    --save_path $SAVE_PATH