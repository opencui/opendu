import argparse
import json
import logging
import os


import shutil
from dataclasses import dataclass, field
from enum import Enum
from os.path import exists, isdir, join
from typing import Dict, Optional, List, Tuple

import evaluate
import numpy as np
import torch
import transformers
from datasets import Dataset, interleave_datasets
from peft import LoraConfig, get_peft_model, TaskType, PrefixTuningConfig

from transformers import (AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, set_seed,
                          DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM)

from opendu.core.config import RauConfig, ModelType
from opendu.core.special_tokens import SpecialTokens
from opendu.finetune.commons import (load_training_dataset)
from opendu.finetune.datacollator import DataCollatorForCausalLM

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

BoolType = Enum("BoolType", ["true", "false"])

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
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
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=256,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset: str = field(
        default="alpaca",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"
        },
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )

    report_to: str = field(
        default=None,
        metadata={"help": "To use wandb or something else for reporting."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="adamw_hf", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    fp16: bool = field(
        default=True, metadata={"help": "Whether or not use fp16 during training."}
    ),
    bf16: bool = field(
        default=False, metadata={"help": "Whether or not use bf16 during training."}
    ),
    debug_dataset: bool = field(
        default=False, metadata={"help": "print out dataset instead"}
    )
    training_mode: str = field(default="skill", metadata={"help": "skill or slot"})
    peft_mode: str = field(default="null", metadata={"help": "lora or prompt-tuning"})
    model_type: str = field(default="gpt", metadata={"help": "gpt or t5, just need to be t2t"})


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
            "if predict_with_generate is set."
        },
    )
    min_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "Minimum number of new tokens to generate."}
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


def get_model(args, peft_config=None):
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    print(f"loading base model {args.model_name_or_path}...")
    model = None
    if ModelType[args.model_type] == ModelType.gpt:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

    if ModelType[args.model_type] == ModelType.t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

    if peft_config is not None:
        print("Using peft instead.")
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False
        # Using torch.bfloat16
        for name, module in model.named_modules():
            if "norm" in name:
                module.to(torch.bfloat16)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        use_fast=True,  # Fast tokenizer giving issues.
        trust_remote_code=args.trust_remote_code,
    )

    special_tokens_dict = dict()
    if tokenizer._pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"

    # We add some special tokens.
    if ModelType[args.model_type] == ModelType.gpt:
        special_tokens_dict['additional_special_tokens'] = SpecialTokens.list()

    # For now, regardless, we always train in multitasks, so no need to add special token.
    #smart_tokenizer_and_embedding_resize(
    #    special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model
    #)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

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
    print(f"trainable params: {trainable_params} || " f"all params: {all_param} || ")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")


@dataclass
class DatasetAdaptor:
    def __init__(self, tokenizer, max_source_length, max_target_length):
        self.padding = True
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, sample: Dataset) -> dict:
        """ Preprocess the dataset. """

        # add prefix to the input for t5
        inputs = sample["input"]

        # tokenize inputs
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(
            text_target=sample["output"], max_length=self.max_target_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def merge_created_datasets(creators, split: str) -> Dataset:
    datasets = []
    for creator in creators:
        dataset = creator.__getitem__(split)
        if dataset is not None:
            datasets.append(dataset)
    # Change to interleave so that we can up sample these datasets
    return interleave_datasets(datasets,  stopping_strategy="all_exhausted")


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


class F1MetricComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def parse_true_false(item):
        if item == "true":
            return 0
        elif item == "false":
            return 1
        else:
            return 2

    @staticmethod
    def postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """ helper function to postprocess text"""
        preds = [F1MetricComputer.parse_true_false(pred.strip()) for pred in preds]
        labels = [F1MetricComputer.parse_true_false(label.strip()) for label in labels]
        print(type(preds))
        return preds, labels

    def __call__(self, eval_preds: transformers.trainer_utils.EvalPrediction):
        metric = evaluate.load("f1")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = F1MetricComputer.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def train():
    # Turn off the evaluation mode, but why?
    RauConfig.get().eval_mode = False

    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    peft_config = None
    if args.peft_mode == "lora":
        peft_config = get_lora_config()
    if args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=8
        )

    # For now, just use the fix path.
    output = "../output"

    training_args.report_to = []

    # Save the things to disk first, for training we keep each module separate.
    # Down the road, we might
    converted_factories = load_training_dataset(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    print("loaded model")
    set_seed(args.seed)
    data_collator = None
    model, tokenizer = get_model(args, peft_config)

    if ModelType[args.model_type] == ModelType.gpt:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )

    if ModelType[args.model_type] == ModelType.t5:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=IGNORE_INDEX,
            pad_to_multiple_of=8
        )

    # this creates a dict.
    # Split train/eval, reduce size
    eval_dataset = merge_created_datasets(converted_factories, "validation")
    train_dataset = merge_created_datasets(converted_factories, "train")

    if ModelType[args.model_type] == ModelType.t5:
        preprocess = DatasetAdaptor(tokenizer, args.source_max_len, args.target_max_len)
        eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=['input', 'output'])
        train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['input', 'output'])

    if ModelType[args.model_type] == ModelType.t5:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=F1MetricComputer(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    if ModelType[args.model_type] == ModelType.gpt:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
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
        trainer.save_model()
        all_metrics.update(metrics)

        # append save the config
        if ModelType[args.model_type] == ModelType.t5:
            shutil.copy("./opendu/finetune/t5.sh", f"{args.output_dir}/")
        else:
            shutil.copy("./opendu/finetune/gpt.sh", f"{args.output_dir}/")
        shutil.copy("./opendu/core/config.py", f"{args.output_dir}/")

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

        # create the soft link so that it is easy for experiments.
        checkpoint_files = os.listdir(trainer.args.output_dir)
        checkpoint_files = [file for file in checkpoint_files if "checkpoint" in file]
        last_checkpoint = os.path.join(trainer.args.output_dir, max(checkpoint_files, key=lambda x: int(x.split("-")[-1])))
        last_path = os.path.join(trainer.args.output_dir, "last")

        if os.path.exists(last_path):
            print(f"remove {last_path}")
            os.remove(last_path)

        check_name = os.path.basename(last_checkpoint)
        os.symlink(check_name, last_path)

        # fix the generate_config.json
        os.makedirs(last_path, exist_ok=True)
        generation_config = os.path.join(last_path, "generation_config.json")

        with open(generation_config, 'r') as json_file:
            config = json.load(json_file)

        # this is what is missing.
        config['decoder_start_token_id'] = 0

        with open(generation_config, 'w') as json_file:
            json.dump(config, json_file, indent=2)



def get_lora_config():
    lora_alpha = 16  # 16
    lora_dropout = 0.1  # 0.1
    lora_rank = 8  # 64
    # There difference choices, and not much explanation.
    # https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
    anyscale_blog = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
    # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    # https://github.com/huggingface/peft/pull/337/files
    hf_flavor = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
    #
    anyscale_code = ["gate_proj", "up_proj", "down_proj"],
    # According to https://github.com/huggingface/peft/issues/334, this should work, but it does not.
    modules_to_save = ["lm_head", "embed_tokens"]  # for llama
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=anyscale_blog)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    # Now we need to create the converters.
    train()
