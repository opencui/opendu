from abc import ABC, abstractmethod, ABCMeta
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Any
import numpy as np
import logging
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from core.prompt import SkillPrompts, OneSlotPrompts
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
)
from datasets import Dataset, concatenate_datasets


from core.annotation import ModuleSchema
from core.embedding import EmbeddingStore
from core.prompt import Prompt
from core.retriever import build_nodes_from_skills, create_index, load_context_retrievers, ContextRetriever
from finetune.commons import AnnotatedExemplar, DatasetFactory, build_nodes_from_dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# This converter is responsible for convert the exemplars in the original dataset into what is needed
# by generation fine-tuning. The assumed the columns are input and output, and we added id for debugging
# purpose.
class Converter(ABC):
    prompt: Prompt

    @abstractmethod
    def __call__(self, item:AnnotatedExemplar, ins: list[str], outs: list[str], ids: list[str]):
        return


class SkillConverter(Converter):
    def __init__(self, retriever: ContextRetriever, func_prompt: Prompt):
        self.prompt = func_prompt
        self.context_retrieve = retriever

    def __call__(self, item: AnnotatedExemplar, ins: list[str], outs: list[str], ids: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        skills, exemplars = self.context_retrieve(item["utterance"])
        # remove the identical exemplar
        exemplars = [exemplar for exemplar in exemplars if exemplars.id != item[':id']]
        input_dict = item.to_dict()
        input_dict["examples"] = exemplars
        input_dict["skills"] = skills
        ins.append(self.prompt(input_dict))
        outs.append(item.owner)
        ids.append(item.id)


#
# This is for extractive slot value understanding.
#
class OneSlotConverter(Converter):
    def __init__(self, module: ModuleSchema, slot_prompt: Prompt):
        self.prompt = slot_prompt
        self.module = module

    def __call__(self, item: AnnotatedExemplar, ins: list[str], outs: list[str], ids: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        input_dict = item.to_dict()
        for slot, value in item.arguments:
            input_dict["name"] = slot
            input_dict["description"] = self.module.slots[slot].description
            ins.append(self.slot_prompt(item))
            outs.append(value)
        return {"input": ins, "output": outs}


# This is needed to determine the intention, intended function or skill


# This converter is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
# This is needed to determine the intention, intended function or skill
class BooleanConverter(Converter):
    def __init__(self, module: ModuleSchema, func_prompt: Prompt):
        self.func_prompt = func_prompt
        self.module = module

    def __call__(self, item: AnnotatedExemplar, ins: list[str], outs: list[str], ids: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        input_dict = item.to_dict()

        ins.append(self.func_prompt(input_dict))
        outs.append(item.owner)

        for slot, value in item.arguments:
            input_dict["name"] = slot
            input_dict["description"] = self.module.slots[slot].description
            ins.append(self.slot_prompt(item))
            outs.append(value)
        return {"input": ins, "output": outs}


@dataclass
class ConvertedFactory(DatasetFactory):
    __metaclass__ = ABCMeta

    def __init__(self, dsf: DatasetFactory, convert: list[Converter]):
        self.domain = dsf.domain
        self.converters: list[Converter] = convert

    def extra_tokens(self):
        return set([token for converter in self.converters for token in converter.prompt.extr_tokens]).to_list()

    def convert(self, item):
        ins = []
        outs = []
        ids =[]
        for convert in self.converters:
            convert(item, ins, outs, ids)
        assert len(ins) == len(outs)
        if len(ins) == len(ids):
            return {"input": ins, "output": outs, "id": ids}
        else:
            return {"input": ins, "output": outs}

    def build(self, split: str) -> Dataset:
        dataset = self.creator.build(split)
        return dataset.map(self.convert, batched=True, remove_columns=dataset.column_names)


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


def get_accelerate_model(args, extra_special_tokens: set[str]):
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
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


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out


def get_dataset(creators, split: str) -> Dataset:
    datasets = []
    for creator in creators:
        dataset = creator.build(split)
        if dataset is not None:
            datasets.append(dataset)
    return concatenate_datasets(datasets)


def make_data_module(data_collator, args, converters) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `utterance`, `output` }

    Available datasets to be selected with `dataset` argument:
        - viggo
    """
    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        eval_dataset = get_dataset(converters, "validation")
    if args.do_train:
        train_dataset = get_dataset(converters, "train")

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train(converted_factories: list[ConvertedFactory]):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))

    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    extra_tokens = set([token for factory in converted_factories for token in factory.extra_tokens()])
    model, tokenizer = get_accelerate_model(args, extra_tokens)

    model.config.use_cache = False
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


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    from finetune.sgd import SGD
    from converter.lug_config import LugConfig
    LugConfig.embedding_device = "cuda"

    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/")]

        # For now, just use the fix path.
    output = "./output"
    retrievers = []
    for factory in factories:
        retrievers.append(load_context_retrievers(factory.domain, f"{output}/index/{factory.tag}"))


    converted_factories = []
    for index, factory in enumerate(factories):
        context_retriever = retrievers[index]
        skill_converter = SkillConverter(context_retriever, SkillPrompts["exampled_prompt_for_skill00"])
        slot_converter = OneSlotConverter(factory.domain, OneSlotPrompts["basic"])
        converted_factory = ConvertedFactory(factory, [skill_converter, slot_converter])

    # Now we need to create the converters.
    train(converted_factories)
