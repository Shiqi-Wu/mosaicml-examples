# src/no_meta_hf.py
from composer.models.huggingface import HuggingFaceModel

class NoMetaHFModel(HuggingFaceModel):
    def get_metadata(self):
        # Don't call super(); this avoids tokenizer.save_pretrained entirely.
        return {}
