import os
import argparse
import numpy as np
from tqdm import tqdm
from streaming import MDSWriter
from transformers import AutoTokenizer

def convert_txt_to_mds(input_txt, output_dir, tokenizer_name, max_length=512, bos_text="", eos_text=""):
    os.makedirs(output_dir, exist_ok=True)
    print("[DEBUG] Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("[DEBUG] Step 2: Tokenizer loaded successfully.")
    tokenizer.model_max_length = int(1e30)

    bos_tokens = tokenizer(bos_text, add_special_tokens=False)['input_ids'] if bos_text else []
    eos_tokens = tokenizer(eos_text, add_special_tokens=False)['input_ids'] if eos_text else []

    columns = {
        "tokens": "bytes",
        "type": "str", 
    }
    print(f"[DEBUG] Step 3: Converting data to MDS format...")
    max_lines = 1000
    with MDSWriter(out=output_dir, columns=columns) as writer:
        with open(input_txt, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if idx >= max_lines:
                    break

                input_ids = tokenizer(line, truncation=True, max_length=512)["input_ids"]
                if not input_ids:
                    continue

                token_bytes = np.array(input_ids, dtype=np.int16).tobytes()
                if len(token_bytes) % 2 != 0:
                    print(f"[WARN] Token byte length not divisible by 2 at line {idx}. Skipped.")
                    continue
                writer.write({
                    "tokens": token_bytes,
                    "type": "int16",
                })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True, help='Path to input .txt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for MDS')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer name or path')
    parser.add_argument('--max_length', type=int, default=512, help='Max token sequence length')
    parser.add_argument('--bos_text', type=str, default='', help='Text to prepend to each sample')
    parser.add_argument('--eos_text', type=str, default='', help='Text to append to each sample')

    args = parser.parse_args()
    convert_txt_to_mds(args.input_txt, args.output_dir, args.tokenizer,
                       args.max_length, args.bos_text, args.eos_text)
