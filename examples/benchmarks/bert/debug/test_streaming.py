# test_stream.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_data import StreamingTextDataset

ds = StreamingTextDataset(
    remote='/mnt/data/dnabert_2_pretrain_full/train',
    local='/tmp/test_dl',
    split=None,
    tokenizer_name='./saved_models/tokenizer_ckpt',
    max_seq_len=128,
    shuffle=False,
    mlm_probability=0.15,
    download=True,
)

print("✅ Dataset constructed. Length:", len(ds))
print("🔁 Sample:", next(iter(ds)))
