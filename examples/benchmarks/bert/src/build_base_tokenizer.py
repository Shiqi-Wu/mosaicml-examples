import os
import json
import argparse
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Split


def build_base_tokenizer(vocab=None) -> PreTrainedTokenizerFast:
    if vocab is None:
        vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[UNK]": 2,
            "[CLS]": 3,
            "A": 4,
            "C": 5,
            "G": 6,
            "T": 7,
        }

    tokenizer_core = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Split(pattern="", behavior="isolated")

    tokenizer_core.post_processor = TemplateProcessing(
        single="$0",
        pair="$A $B",
        special_tokens=[("[PAD]", vocab["[PAD]"])]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        unk_token="N",
        pad_token="[PAD]"
    )

    return tokenizer

def save_tokenizer(tokenizer: PreTrainedTokenizerFast, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Tokenizer saved to: {save_path}")

def test_tokenizer(tokenizer: PreTrainedTokenizerFast):
    example = "AGTCCGTAN"
    tokens = tokenizer(example, add_special_tokens=False)
    print(f"Input: {example}")
    print(f"Encoded: {tokens['input_ids']}")
    print(f"Decoded: {tokenizer.decode(tokens['input_ids'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save a base-level character tokenizer (A/C/G/T/N).")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the tokenizer files.")
    args = parser.parse_args()

    tokenizer = build_base_tokenizer()
    test_tokenizer(tokenizer)
    save_tokenizer(tokenizer, args.save_path)
