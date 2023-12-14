import argparse
import copy
import json
import logging
import os

import shutil
from dataclasses import dataclass, field
from os.path import exists, isdir, join
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, set_seed)
from opencui.core.embedding import EmbeddingStore
from opencui.core.prompt import (ExtractiveSlotPrompts, NliPrompts)
from opencui.core.retriever import (build_desc_index, load_context_retrievers)
from opencui.finetune.commons import (MappedDatasetDict, build_dataset_index, collect_slot_values, JsonDatasetFactory,
                                      OneSlotExtractConverter, PromptedFactory, NliConverter)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


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
        default="none",
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
    debug_dataset: bool = field(
        default=False, metadata={"help": "print out dataset instead"}
    )
    training_mode: str = field(default="skill", metadata={"help": "skill or slot"})


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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "lm_head",
        ],
    )


def get_accelerate_model(args, extra_special_tokens: set[str], peft_config=None):
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    print(f"loading base model {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
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
        special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model
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
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{example['input']}" for example in instances]
        targets = [
            f"{example['output']} {self.tokenizer.eos_token}" for example in instances
        ]

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
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
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
        eval_dataset = merge_created_datasets(converters, "validation")
    if args.do_train:
        train_dataset = merge_created_datasets(converters, "train")

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


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


def train(peft_config=None):
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
        print("Detected that training was already completed!")

    extra_tokens = set(
        [token for factory in converted_factories for token in factory.extra_tokens()]
    )

    model, tokenizer = get_accelerate_model(args, extra_tokens, peft_config)

    print("loaded model")
    set_seed(args.seed)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    data_module = make_data_module(
        data_collator=data_collator, args=args, converters=converted_factories
    )
    print("prepared data.")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
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
        all_metrics.update(metrics)

        # append save the config
        shutil.copy("./opencui/finetune/generation.sh", f"{args.output_dir}/")
        shutil.copy("./opencui/core/config.py", f"{args.output_dir}/")

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
        prediction_output = trainer.predict(
            test_dataset=data_module["predict_dataset"], metric_key_prefix="predict"
        )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = (
                    predictions[i].replace(example["input"], "").strip()
                )
                fout.write(json.dumps(example) + "\n")
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
        JsonDatasetFactory("./datasets/sgd/", "sgd", "prompted."),
    ]
    return factories


def build_extractive_slot_factory():
    factories = [
        JsonDatasetFactory("./datasets/sgd/", "sgd"),
    ]
    converted_factories = []
    for index, factory in enumerate(factories):
        entity_values = collect_slot_values(factory.__getitem__("train"))
        slot_converter = OneSlotExtractConverter(
            factory.schema, ExtractiveSlotPrompts[LugConfig.slot_prompt], entity_values
        )
        converted_factories.append(PromptedFactory(factory, [slot_converter]))

    return converted_factories


def build_nli_factory():
    # Here we assume the raw input is sentence, focus and label (positive, negative and neutral)
    semeval2016 = load_dataset("glue", "mnli")
    factories = [MappedDatasetDict(semeval2016, "validation_matched", "validation_mismatched")]
    converted_factories = []
    for index, factory in enumerate(factories):
        converter = NliConverter(NliPrompts[LugConfig.nli_prompt])
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

    from opencui.finetune.sgd import SGD
    from opencui.core.config import LugConfig

    LugConfig.embedding_device = "cuda"

    # Now we need to create the converters.
    train(get_lora_config())
