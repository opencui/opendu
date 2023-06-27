import gin
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
import sys
from transformers import AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig

from string import Template


# this is dependency
# pip install -q peft transformers datasets gin-config
# pip install -U sentencepiece tokenizers (need to update to 0.13.3)



# We need to define all the options in gin.
# This is the prompt tuning, the goal it to make it easy to serve multiple task with one base model.

# parse the config so that


@gin.configurable
class PromptTuner:
    def __init__(self, model_name_or_path, dataset_format, dataset_name):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare for the basic dataset.
        dataset = load_dataset(dataset_format, dataset_name)
        classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
        print(classes)
        self.dataset = dataset.map(
            lambda x: {"text_label": [classes[label] for label in x["Label"]]},
            batched=True,
            num_proc=1,
        )
        self.target_max_length = max([len(self.tokenizer(class_label)["input_ids"]) for class_label in classes])

        self.config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=8,
        )

@gin.configurable
class RaftPreprocessor:
    def __init__(self, tokenizer, text_column, label_column, max_length):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __call__(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"{self.text_column} : {x} Label : " for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tuner.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tuner.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

@gin.configurable
class Optimizer:
    def __init__(self, train_dataset, eval_dataset, lr, num_epochs, batch_size):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        self.eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * num_epochs),
        )
        self.device = "cpu"

    def tune(self, rmodel, tokenizer):
        model = rmodel.to(self.device)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                self.lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(self.eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")




# Substitute value of x in above template
if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    tuner = PromptTuner()

    preprocess_function = RaftPreprocessor(tuner.tokenizer)

    processed_datasets = tuner.dataset.map(
        preprocess_function,
        batched = True,
        num_proc = 1,
        remove_columns = tuner.dataset["train"].column_names,
        load_from_cache_file = False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["train"]

    model = AutoModelForCausalLM.from_pretrained(tuner.model_name_or_path)
    model = get_peft_model(model, tuner.config)
    print(model.print_trainable_parameters())

    optimizer = Optimizer()
    optimizer.tune(model, tuner.tokenizer)

    # Figure out how to push model to huggingface.
    peft_model_id = "opencui/test_PROMPT_TUNING_CAUSAL_LM"
    model.push_to_hub("opencui/test_PROMPT_TUNING_CAUSAL_LM", use_auth_token=True)

