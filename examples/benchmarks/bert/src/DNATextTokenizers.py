from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Union, Optional
import torch
import os
import json
import re


class DNATextUnifiedTokenizer(PreTrainedTokenizerBase):
    def __init__(self, text_tokenizer, dna_tokenizer, dna_offset=30528):
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.dna_tokenizer = dna_tokenizer
        self.dna_offset = dna_offset

        self.pad_token_id = text_tokenizer.pad_token_id
        self.mask_token_id = text_tokenizer.mask_token_id
        self.cls_token_id = text_tokenizer.cls_token_id
        self.sep_token_id = text_tokenizer.sep_token_id

        # print("dna_offset:", self.dna_offset)
        # print("text vocab size:", text_tokenizer.vocab_size)
        # print("dna vocab size:", dna_tokenizer.vocab_size)
        self.vocab_size = self.dna_offset + dna_tokenizer.vocab_size

    def __call__(self,
                 text: str,
                 padding=True,
                 truncation=True,
                 max_length=None,
                 return_tensors="pt",
                 **kwargs) -> Dict[str, torch.Tensor]:
        pattern = re.compile(r'<dna>(.*?)<dna>')
        parts, last = [], 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > last:
                parts.append(("text", text[last:start]))
            parts.append(("dna", match.group(1)))
            last = end
        if last < len(text):
            parts.append(("text", text[last:]))

        input_ids = []
        modality_mask = []
        attention_mask = []

        for mod, seg in parts:
            if mod == "text":
                ids = self.text_tokenizer(seg, add_special_tokens=False)["input_ids"]
                mods = [0] * len(ids)
            elif mod == "dna":
                raw_ids = self.dna_tokenizer(seg, add_special_tokens=False)["input_ids"]
                ids = [self.mask_token_id if i == self.dna_tokenizer.mask_token_id else i + self.dna_offset for i in raw_ids]
                mods = [1] * len(ids)
            input_ids.extend(ids)
            modality_mask.extend(mods)
            attention_mask.extend([1] * len(ids))

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "modality_mask": torch.tensor([modality_mask], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int): ids, single = [ids], True
        else: single = False
        out = [self.text_tokenizer.convert_ids_to_tokens(i) if i < self.dna_offset else self.dna_tokenizer.convert_ids_to_tokens(i - self.dna_offset) for i in ids]
        return out[0] if single else out

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str): tokens, single = [tokens], True
        else: single = False
        out = [self.text_tokenizer.convert_tokens_to_ids(t) for t in tokens]
        return out[0] if single else out

    def decode(self, input_ids: List[int], skip_special_tokens=False):
        if skip_special_tokens:
            input_ids = [i for i in input_ids if i != self.pad_token_id]
        tokens = self.convert_ids_to_tokens(input_ids)
        return " ".join(tokens)

    def decode_segments(self, input_ids_batch: torch.Tensor, skip_special_tokens=True) -> str:
        all_ids = []
        for ids in input_ids_batch:
            ids = ids.tolist()
            if skip_special_tokens:
                ids = [i for i in ids if i != self.pad_token_id]
            all_ids.extend(ids)
        return self.decode(all_ids, skip_special_tokens=False)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.text_tokenizer.save_pretrained(os.path.join(save_directory, "text"))
        self.dna_tokenizer.save_pretrained(os.path.join(save_directory, "dna"))
        config = {"dna_offset": self.dna_offset}
        with open(os.path.join(save_directory, "dna_text_tokenizer_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        from transformers import AutoTokenizer
        with open(os.path.join(path, "dna_text_tokenizer_config.json")) as f:
            config = json.load(f)
        text_tok = AutoTokenizer.from_pretrained(os.path.join(path, "text"))
        dna_tok = AutoTokenizer.from_pretrained(os.path.join(path, "dna"))
        return cls(text_tok, dna_tok, dna_offset=config["dna_offset"])

    def __len__(self):
        return self.vocab_size



if __name__ == "__main__":
    from transformers import AutoTokenizer

    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dna_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    tokenizer = DNATextUnifiedTokenizer(text_tokenizer, dna_tokenizer, dna_offset=30528)

    input_text = "TP53 is important: <dna>ATGCGT<dna>."
    output = tokenizer(input_text)
    print("👉 input_ids:\n", output["input_ids"])
    print("👉 modality_mask:\n", output["modality_mask"])
    print("👉 decoded:\n", tokenizer.decode_segments(output["input_ids"]))

    tokenizer.save_pretrained("./saved_models/tokenizer_ckpt")
    tokenizer2 = DNATextUnifiedTokenizer.from_pretrained("./saved_models/tokenizer_ckpt")
    output2 = tokenizer2(input_text)
    print("✅ reload ok, decoded again:\n", tokenizer2.decode_segments(output2["input_ids"]))
    print("len(tokenizer2):", len(tokenizer2))
