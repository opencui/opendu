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

from converter.lug_config import LugConfig
from core.embedding import EmbeddingStore
from core.retriever import build_desc_index, HybridRetriever, load_retrievers, CombinedRetriever
from finetune.commons import build_dataset_index
from finetune.sgd import SGD


def compute_k(dataset: Dataset, retriever: CombinedRetriever):
    counts = [0, 0]
    for item in dataset:
        skills, exemplars = retriever.search(item["utterance"])
        if item["owner"] == "NONE":
            continue

        intents = set([skill["name"] for skill in skills])
        counts[0] += 1
        if item["owner"] in intents:
            counts[1] += 1
        else:
            print(f">>>>>{item}: \n +++++ {skills}\n --------{exemplars} \n\n")

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

    LugConfig.embedding_device = "cuda"

    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/")]

    # For now, just use the fix path.
    output = "./output"

    build_index = True
    if build_index:
        for factory in factories:
            build_desc_index(factory.domain, f"{output}/index/{factory.tag}", EmbeddingStore.for_description())
            build_dataset_index(factory.build("train"), f"{output}/index/{factory.tag}", EmbeddingStore.for_exemplar())

    retrievers = []
    for factory in factories:
        retrievers.append(load_retrievers(factory.domain, f"{output}/index/{factory.tag}"))

    #searcher = retrievers[0]
    #nodes = searcher.search("i want to go out to eat somewhere")
    for index in range(len(factories)):
        factory = factories[index]
        searcher = retrievers[index]
        ds = factory.build("train")
        print(compute_k(ds, searcher))
