import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np
import logging
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
)

from datasets import Dataset, concatenate_datasets

from core.embedding import EmbeddingStore
from core.retriever import build_desc_index, HybridRetriever, load_retrievers
from finetune.commons import build_dataset_index
from finetune.sgd import SGD


def compute_k(dataset: Dataset, output: str, tag: str, topk: int = 3):
    retriever = HybridRetriever(output, tag, topk=8)
    counts = [0, 0]
    for item in dataset:
        nodes = retriever.retrieve(item["utterance"])
        intents = set()
        lintents = []
        for result in nodes:
            intent = result.node.metadata["owner"]
            if intent not in intents:
                intents.add(intent)
                lintents.append(intent)
            if len(lintents) >= topk:
                break
        counts[0] += 1
        if item["owner"] in lintents[0:topk]:
            counts[1] += 1

    return counts


def compute_hits(dataset: Dataset, output: str, topk: int):
    retriever = HybridRetriever(output, "desc", topk=topk)
    counts = [0, 0]
    for item in dataset:
        nodes = retriever.retrieve(item["utterance"])
        intents = {result.node.metadata["owner"] for result in nodes}
        counts[0] += 1
        name = item["owner"]
        if name in intents or name == "NONE":
            counts[1] += 1
        else:
            print(f'{name}:{item["utterance"]} not in {intents}')

    return counts


#
# It is really import that we get the hyperparameter right. For fine-tune the generator in the RAG,
# we need to make sure the prompt template can be instantiated to meet the certain criteria.
# In particular, we need to find some constant in terms of how many function and exemplars do we need
# include to have a high enough probability to have correct function included in the context.
#
if __name__ == "__main__":
    # The first thing is to create the schema and create the datasets of annotated exemplars.
    # Then create the index for both descriptions and exemplars on training split.
    # Then define the prompt.
    # Then figure out the good k using validation split. These Ks will be used for inference and training.
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/")]

    # For now, just use the fix path.
    output = "./output"

    for factory in factories:
        build_desc_index(factory.domain, f"{output}/index/{factory.tag}", EmbeddingStore.for_description())
        build_dataset_index(factory.build("train"), f"{output}/index/{factory.tag}", EmbeddingStore.for_exemplar())

    retrievers_list = []
    for factor in factories:
        retrievers_list.append(load_retrievers(f"{output}/index/{factory.tag}"))

