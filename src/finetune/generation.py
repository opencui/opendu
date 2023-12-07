import re
from abc import ABC, abstractmethod, ABCMeta
import copy
import json
import os
import shutil
from enum import Enum
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import random
from typing import Optional, Dict, Sequence
import numpy as np
import logging
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from core.prompt import MulticlassSkillPrompts, ExtractivePrompts, BinarySkillPrompts, LayeredPrompts, NLIPrompts
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
)
from datasets import Dataset, concatenate_datasets, load_dataset

from core.annotation import Schema, Exemplar, ListRecognizer
from core.embedding import EmbeddingStore
from core.prompt import Prompt
from core.retriever import load_context_retrievers, ContextRetriever, build_desc_index
from finetune.commons import AnnotatedExemplar, DatasetFactory, build_dataset_index, collect_slot_values
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# This inference is responsible for convert the exemplars in the original dataset into what is needed
# by generation fine-tuning. The assumed the columns are input and output, and we added id for debugging
# purpose.
class TrainConverter(ABC):
    prompt: Prompt

    @abstractmethod
    def __call__(self, item: AnnotatedExemplar, ins: list[str], outs: list[str]):
        return


# The slot converter need to have access to entities.
class SlotExtractConverter(TrainConverter, ABC):
    entities: dict[str, re.Pattern]


# This is needed to determine the intention, intended function or skill
# https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
class SkillTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever):
        self.prompt = MulticlassSkillPrompts[LugConfig.skill_prompt]
        self.context_retrieve = retriever

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)
            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch['id'][idx]]
            exemplars = [Exemplar(owner=node.metadata["owner"], template=node.text) for node in nodes]
            owner = batch["owner"][idx]

            # How can we reduce the need for

            neg_owners = [node.metadata["owner"] for node in nodes if node.metadata["owner"] != owner]

            # randomly filter one neg skills and exemplars
            if len(neg_owners) != 0:
                neg_owner = random.choice(neg_owners)
                rm_neg_exemplars = [exemplar for exemplar in exemplars if exemplar.owner != neg_owner]
                rm_neg_skills = [skill for skill in skills if skill["name"] != neg_owner]

                # Without exemplars.
                random.shuffle(rm_neg_skills)
                input_dict = {"utterance": utterance, "examples": [], "skills": rm_neg_skills}
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner)}</s>")

                # With exemplars.
                if len(rm_neg_exemplars) != 0:
                    random.shuffle(rm_neg_exemplars)
                    input_dict = {"utterance": utterance, "examples": rm_neg_exemplars, "skills": rm_neg_skills}
                    ins.append(self.prompt(input_dict))
                    outs.append(f"{json.dumps(owner)}</s>")

            # Try to filter the pos skills and exemplars
            rm_pos_exemplars = [exemplar for exemplar in exemplars if exemplar.owner != owner]
            rm_pos_skills = [skill for skill in skills if skill["name"] != owner]

            random.shuffle(rm_pos_skills)
            input_dict = {"utterance": utterance, "examples": [], "skills": rm_pos_skills}
            ins.append(self.prompt(input_dict))
            outs.append(f"{json.dumps(None)}</s>")

            if len(rm_pos_exemplars) != 0:
                random.shuffle(rm_pos_exemplars)
                input_dict = {"utterance": utterance, "examples": rm_pos_exemplars, "skills": rm_pos_skills}
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(None)}</s>")


class OneSkillTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever):
        self.prompt = BinarySkillPrompts[LugConfig.skill_prompt]
        self.context_retrieve = retriever
        self.neg_k = 1

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)
            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch['id'][idx]]
            exemplars = [Exemplar(owner=node.metadata["owner"], template=node.text) for node in nodes]
            owner = batch["owner"][idx]

            skill_map = {}

            # for the skills
            for skill in skills:
                input_dict = {"utterance": utterance, "examples": [], "skill": skill}
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner == skill['name'])}</s>")
                skill_map[skill["name"]] = skill

            for o_exemplar in exemplars:
                target = o_exemplar.owner
                # Try not to have more than two examples.
                exemplar_dicts = [
                    {"template": exemplar.template, "target": target, "decision": target == exemplar.owner}
                    for exemplar in exemplars]

                input_dict = {"utterance": utterance, "examples": exemplar_dicts, "skill": skill_map[target]}
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner == target)}</s>")


# For this one, we first use example based prediction, and then description based prediction.
class LayeredTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever):
        self.desc_prompt = LayeredPrompts[LugConfig.skill_prompt][0]
        self.example_prompt = LayeredPrompts[LugConfig.skill_prompt][1]
        self.context_retrieve = retriever
        self.neg_k = 1

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)
            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch['id'][idx]]
            exemplars = [Exemplar(owner=node.metadata["owner"], template=node.text) for node in nodes]
            owner = batch["owner"][idx]

            skill_map = {}

            # for the skills
            for skill in skills:
                input_dict = {"utterance": utterance, "skill": skill}
                ins.append(self.desc_prompt(input_dict))
                outs.append(f"{json.dumps(owner == skill['name'])}</s>")
                skill_map[skill["name"]] = skill

            for exemplar in exemplars:
                target = exemplar.owner
                # Try not to have more than two examples.
                input_dict = {"utterance": utterance, "template": exemplar.template}
                ins.append(self.example_prompt(input_dict))
                outs.append(f"{json.dumps(owner == target)}</s>")


#
# This is for extractive slot value understanding.
# For now, we only get positive example.
class OneSlotExtractConverter(SlotExtractConverter):
    def __init__(self, module: Schema, slot_prompt: Prompt, entities):
        self.prompt = slot_prompt
        self.module = module
        self.include_negative = True
        # First try to be efficient.
        self.entities = entities
        self.patterns = {}
        for key, values in entities.items():
            strings_to_check = list(values)
            pattern = re.compile('|'.join(map(re.escape, strings_to_check)))
            self.patterns[key] = pattern

    @staticmethod
    def format_value(key, value=None):
        return f"{json.dumps(value)}</s>"

    def add_one_negative(self, slot_name, small_value_set):
        if slot_name not in self.entities:
            return
        
        picked = None
        candidates = self.entities[slot_name]

        while picked in small_value_set:
            picked = random.choice(candidates)

        if picked is not None:
            small_value_set.add(picked)

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, sarguments in enumerate(batch["arguments"]):
            arguments = eval(sarguments)
            utterance = batch["utterance"][idx]
            owner = batch["owner"][idx]
            for slot_label in self.module.skills[owner]["slots"]:
                slot = self.module.slots[slot_label]
                slot_name = slot["name"]

                # Now we need to select the value from entities
                # In addition to the true value, the best should be of the same type and
                # also the occurs in the utterance but not the value.
                values = set(ListRecognizer.find_matches(self.patterns, slot_name, utterance))
                # Most likely we do not need to add the negatives.
                # self.add_one_negative(slot_label, values)
                input_dict = {"utterance": utterance}
                input_dict.update(slot)
                if slot_name in arguments:
                    value = arguments[slot_name]
                    # First without values. We assume that value is
                    input_dict["values"] = []
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(self.format_value(slot_name, arguments[slot_name][0]))
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                    # then with values.
                    input_dict["values"] = values
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(self.format_value(slot_name, arguments[slot_name][0]))
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                else:
                    input_dict["values"] = []
                    if self.include_negative:
                        ins.append(self.prompt(input_dict))
                        outs.append(self.format_value(slot_name, None))


# We need to handle many different use case here: premise is what user said, and hypothesis is what we want to know.
class NliConverter(TrainConverter, ABC):
    def __init__(self, prompt):
        self.prompt = prompt
        self.labels = ["entailment", "neutral", "contradiction"]

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, premise in enumerate(batch["premise"]):
            hypothesis = batch["hypothesis"][idx]
            label = self.labels[int(batch["label"][idx])]
            input_dict = {"premise": premise, "hypothesis": hypothesis}
            ins.append(self.prompt(input_dict))
            outs.append(f"{label}</s>")


def skill_converter(retriever: ContextRetriever):
    if LugConfig.skill_mode == "binary":
        return OneSkillTrainConverter(retriever)
    if LugConfig.skill_mode == "multiclass":
        return SkillTrainConverter(retriever)
    if LugConfig.skill_mode == "simple":
        return LayeredTrainConverter(retriever)


# This inference is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
# This is needed to determine the intention, intended function or skill
# class BooleanConverter
@dataclass
class PromptedFactory(DatasetFactory):
    __metaclass__ = ABCMeta
    skill_columns = ["id", "utterance", "template", "owner", "arguments", "expectations"]

    def __init__(self, dsf: DatasetFactory, convert: list[TrainConverter], unused_columns=skill_columns):
        self.creator = dsf
        self.converters: list[TrainConverter] = convert
        self.columns = unused_columns

    def extra_tokens(self):
        return list(set([token for converter in self.converters for token in converter.prompt.extra_tokens]))

    def convert_one(self, item):
        ins = []
        outs = []
        for convert in self.converters:
            convert(item, ins, outs)
        assert len(ins) == len(outs)
        return {"input": ins, "output": outs}

    def __getitem__(self, split: str) -> Dataset:
        dataset = self.creator[split]
        return dataset.map(self.convert_one, batched=True, remove_columns=self.columns)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )

    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=16, metadata={
        "help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={
        "help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False,
                                        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={
        "help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True,
                                         metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={
        "help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10,
                               metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={
        "help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40,
                                  metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    debug_dataset: bool = field(default=False, metadata={"help": 'print out dataset instead'})
    training_mode: str = field(default='skill', metadata={"help": 'skill or slot'})


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def get_lora_config():
    lora_alpha = 16  # 16
    lora_dropout = 0.05  # 0.1
    lora_rank = 8  # 64

    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head']
    )


def get_accelerate_model(args, extra_special_tokens: set[str], peft_config=None):
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    print(f'loading base model {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16
    )

    if peft_config is not None:
        print("Using lora instead.")
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        # Do not know what this actually does.
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(torch.bfloat16)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        use_fast=True,  # Fast tokenizer giving issues.
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer._pad_token is None:
        DEFAULT_PAD_TOKEN = "[PAD]"
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)

    special_tokens_dict.update(additional_special_tokens=list(extra_special_tokens))

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )

    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{example['input']}" for example in instances]
        targets = [f"{example['output']} {self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def merge_created_datasets(creators, split: str) -> Dataset:
    datasets = []
    for creator in creators:
        dataset = creator.__getitem__(split)
        if dataset is not None:
            datasets.append(dataset)
    return concatenate_datasets(datasets).shuffle(seed=42)


def make_data_module(data_collator, args, converters) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `utterance`, `output` }

    Available datasets to be selected with `dataset` argument:
        - viggo
    """
    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        eval_dataset = merge_created_datasets(converters, "test")
    if args.do_train:
        train_dataset = merge_created_datasets(converters, "train")

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train(factories: list[DatasetFactory], peft_config=None):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))

    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # append the training mode
    args.output_dir = f"{args.output_dir}/{args.training_mode}"
    if args.training_mode == "skill":
        args.output_dir = f"{args.output_dir}/{LugConfig.skill_mode}"

    # Copy the configure file over so that we know which is which.
    try:
        os.makedirs(args.output_dir, exist_ok = True)
    except OSError as error:
        print("Directory '%s' can not be created" % args.output_dir)

    shutil.copy("./finetune/generation.sh", f"{args.output_dir}/")
    shutil.copy("./core/config.py", f"{args.output_dir}/")

    # For now, just use the fix path.
    output = "../output"

    # Save the things to disk first, for training we keep each module separate.
    # Down the road, we might
    converted_factories = []
    if args.training_mode == "skill":
        converted_factories = build_skill_factory(output)
    if args.training_mode == "extractive_slot":
        converted_factories = build_extractive_slot_factory()
    if args.training_mode == "nli":
        converted_factories = build_nli_factory()

    assert len(converted_factories) != 0

    # If we debug dataset, we do not train.
    if args.debug_dataset:
        for factory in converted_factories:
            ds = factory.__getitem__("train")
            count = [0, 0]
            for item in ds:
                if item["output"] == "null</s>":
                    count[0] += 1
                else:
                    count[1] += 1
                print(json.dumps(item, indent=2))
            print(count)
        exit(0)



    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    extra_tokens = set([token for factory in converted_factories for token in factory.extra_tokens()])

    model, tokenizer = get_accelerate_model(args, extra_tokens, peft_config)

    print('loaded model')
    set_seed(args.seed)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    data_module = make_data_module(data_collator=data_collator, args=args, converters=converted_factories)
    print("prepared data.")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently, adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


# Here we create the dataset factory for skills
def build_skill_factory(output):
    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/"),
    ]
    # Save the things to disk first, for training we keep each module separate.
    # Down the road, we might
    for factory in factories:
        build_desc_index(factory.tag, factory.schema, f"{output}/index/{factory.tag}",
                         EmbeddingStore.for_description())
        build_dataset_index(factory.tag, factory["train"], f"{output}/index/{factory.tag}",
                            EmbeddingStore.for_exemplar())

    retrievers = []
    for factory in factories:
        retrievers.append(load_context_retrievers({factory.tag: factory.schema}, f"{output}/index/{factory.tag}"))

    converted_factories = []
    for index, factory in enumerate(factories):
        context_retriever = retrievers[index]
        converted_factories.append(PromptedFactory(factory, [skill_converter(context_retriever)]))

    return converted_factories


def build_extractive_slot_factory():
    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/"),
    ]
    converted_factories = []
    for index, factory in enumerate(factories):
        entity_values = collect_slot_values(factory.__getitem__("train"))
        slot_converter = OneSlotExtractConverter(
            factory.schema, ExtractivePrompts[LugConfig.slot_prompt], entity_values)
        converted_factories.append(PromptedFactory(factory, [slot_converter]))

    return converted_factories


def build_nli_factory():
    # Here we assume the raw input is sentence, focus and label (positive, negative and neutral)
    semeval2016 = load_dataset("multi-nli")
    factories = [semeval2016]

    converted_factories = []
    for index, factory in enumerate(factories):
        converter = NliConverter(NLIPrompts[LugConfig.nli_prompt])
        converted_factories.append(PromptedFactory(factory, [converter], []))
    return converted_factories


def print_factories(factories):
    for factory in factories:
        ds = factory.__getitem__("train")
        count = 0
        for item in ds:
            print(item)
            count += 1
        print(f"There are {count} instances")



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    from finetune.sgd import SGD
    from core.config import LugConfig

    LugConfig.embedding_device = "cuda"

    # Now we need to create the converters.
    train(get_lora_config())
