# Implements a Embedder for DNA and text sequences using Mosaic BERT-like models.

import os
import sys
from typing import Optional, Union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bert_layers as bert_layers_module
import configuration_bert as configuration_bert_module
import transformers
from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

def create_deepembed_mosaic_bert(
    pretrained_model_name: str = 'bert-base-uncased',
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None
):

    if not model_config:
        mocdel_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "mosaic-bert-base"

    config = configuration_bert_module.BertConfig.from_pretrained(
        pretrained_model_name, **model_config)

    if config.vocab_size % 8 != 0:
        config.vocab_size = (config.vocab_size // 8 + 1) * 8
    
    if pretrained_checkpoint:
        model = bert_layers_module.DeepEmbed_DNABertForMaskedLM.from_pretrained(
            pretrained_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        )
    else:
        model = bert_layers_module.DeepEmbed_DNABertForMaskedLM(config=config)


    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e_auto:
            logger.warning(f"AutoTokenizer loading failed: {e_auto}")

        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        except Exception as e_fast:
            logger.error(f"PreTrainedTokenizerFast loading also failed: {e_fast}")
            raise ValueError(f"Cannot load tokenizer from {tokenizer_name}")
    else:
        tokenizer = None
    
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
