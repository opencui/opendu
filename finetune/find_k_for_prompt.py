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

from finetune.sgd import SGD

#
# It is really import that we get the hyperparameter right. For fine-tune the generator in the RAG,
# we need to make sure the prompt template can be instantiated to meet the certain criteria.
# In particular, we need to find some constant in terms of how many function and exemplars do we need
# include to have a high enough probability to have correct function included in the context.
#
if __name__ == "__main__":
    # The first thing is to create the schema. create the datasets of exemplars.
    # Then create the index for both descriptions and exemplars on training split.
    # Then define the prompt.
    # Then figure out the good k using validation split. These Ks will be used for inference and training.
    sgd = SGD()

    exit()