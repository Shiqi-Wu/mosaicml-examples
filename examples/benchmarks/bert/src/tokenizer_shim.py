# src/tokenizer_shim.py
import os, json

class TokenizerShim:
    """
    Minimal tokenizer surface so dataloaders can call encode/decode AND
    Composer won't choke if it ever tries to save.
    """
    def __init__(self, vocab=None, specials=None, model_max_length=1024):
        self.vocab = vocab or []
        self.specials = specials or {}
        self.model_max_length = model_max_length
        self.padding_side = "right"
        self.truncation_side = "right"

    # If your dataloader uses these, keep them; else safe no-ops.
    def encode(self, s):  # customize to your needs
        return [0] * min(len(s), self.model_max_length)
    def decode(self, ids):
        return ""

    # Composer might never call this now, but make it safe anyway.
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        paths = []
        p = os.path.join(save_directory, "tokenizer_config.json")
        with open(p, "w") as f:
            json.dump({
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
            }, f); paths.append(p)

        p = os.path.join(save_directory, "special_tokens_map.json")
        with open(p, "w") as f:
            json.dump(self.specials, f); paths.append(p)

        p = os.path.join(save_directory, "vocab.txt")
        with open(p, "w") as f:
            f.write("\n".join(self.vocab)); paths.append(p)

        return paths
