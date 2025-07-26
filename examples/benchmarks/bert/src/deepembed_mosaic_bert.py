# Implements a Embedder for DNA and text sequences using Mosaic BERT-like models.

import os
import sys
from typing import Optional, Union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
import bert_layers as bert_layers_module
from dna_text_biembedder import DNABertConfig
from DNATextTokenizers import DNATextUnifiedTokenizer
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
    pretrained_checkpoint: Optional[str] = None,
    init_embeddings_dna2base: Optional[str] = None,
    base_checkpoint: Optional[str] = None,
):

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "mosaic-bert-base"

    bert_base_config = configuration_bert_module.BertConfig.from_pretrained(pretrained_model_name)
    full_config_dict = {**bert_base_config.to_dict(), **model_config}
    print(f"Full config dict: {full_config_dict}")
    config =  DNABertConfig(**full_config_dict)

    if config.vocab_size % 8 != 0:
        config.vocab_size = (config.vocab_size // 8 + 1) * 8
    
    if pretrained_checkpoint:
        model = bert_layers_module.DeepEmbed_DNABertForMaskedLM.from_composer(
            pretrained_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        )
    else:
        if base_checkpoint:
            base_model = bert_layers_module.BertForMaskedLM.from_composer(
            pretrained_checkpoint=base_checkpoint, config=config)
            model = bert_layers_module.DeepEmbed_DNABertForMaskedLM(config=config)
            model.bert.embeddings.word_embeddings.weight.data.copy_(
                base_model.bert.embeddings.word_embeddings.weight.data
            )
            model.bert.encoder.load_state_dict(base_model.bert.encoder.state_dict())
            del base_model
        else:
            model = bert_layers_module.DeepEmbed_DNABertForMaskedLM(config=config)

    init_embeddings_dna2base = torch.load(init_embeddings_dna2base) if init_embeddings_dna2base else None
    if init_embeddings_dna2base is not None:
        model.bert.embeddings.dna_embedder.load_base_embeddings(init_embeddings_dna2base)
    config.vocab_size += config.dna_vocab_size


    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if tokenizer_name is not None:
        tokenizer = DNATextUnifiedTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        LanguageCrossEntropy(ignore_index=-100,
                             vocab_size=model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
