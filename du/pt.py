import gin
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
import sys
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset


# this is dependency
# pip install -q peft transformers datasets gin-config
# pip install -U sentencepiece tokenizers (need to update to 0.13.3)


# We need to define all the options in gin.
# This is the prompt tuning, the goal it to make it easy to serve multiple task with one base model.


@gin.configurable
class PromptTuner:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=8,
        )


@gin.configurable
class OpencuiIntent:
    def __init__(self, dataset_format, dataset_name):
        self.dataset_format = dataset_format
        self.dataset_name = dataset_name

    def __call__(self):
        # Prepare for the basic dataset.
        rdataset = load_dataset(self.dataset_format, self.dataset_name)
        classes = [k.replace("_", " ") for k in rdataset["train"].features["Label"].names]
        print(classes)
        return rdataset.map(
            lambda x: {"text_label": [classes[label] for label in x["Label"]], "text": [f"{'Tweet '} : {y} Label : " for y in x["Tweet text"]]},
            batched=True,
            num_proc=1,
        )


@gin.configurable
class Raft:
    def __init__(self, dataset_format, dataset_name):
        self.dataset_format = dataset_format
        self.dataset_name = dataset_name

    def __call__(self):
        # Prepare for the basic dataset.
        raw_dataset = load_dataset(self.dataset_format, self.dataset_name)
        classes = [k.replace("_", " ") for k in raw_dataset["train"].features["Label"].names]
        print(classes)
        return raw_dataset.map(
            lambda x: {"output": [classes[label] for label in x["Label"]], "input": [f"{'Tweet '} : {y} Label : " for y in x["Tweet text"]]},
            batched=True,
            num_proc=1,
        )


@gin.configurable
class T2TPreprocessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.text_column = "input"
        self.label_column = "output"
        self.max_length = max_length

    def __call__(self, examples):
        # Tokenize the input text and labels.
        # For each example in a batch, pad the labels with the tokenizers pad_token_id.
        # Concatenate the input text and labels into the model_inputs.
        # Create a separate attention mask for labels and model_inputs.
        # Loop through each example in the batch again to pad the input ids, labels, and attention
        #    mask to the max_length and convert them to PyTorch tensors.
        batch_size = len(examples[self.text_column])
        inputs = examples[self.text_column]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
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
    def __init__(self, model, train_dataset, eval_dataset, lr, num_epochs, batch_size):
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
        self.device = "cuda"
        self.model = model

    def tune(self, tokenizer):
        model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

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


@gin.configurable
class Trainer:
    def __init__(self):
        self.tuner = PromptTuner()
        self.model = AutoModelForCausalLM.from_pretrained(self.tuner.model_name_or_path)

    def train(self):
        # prepare the dataset.
        build_dataset = Raft()
        dataset = build_dataset()
        preprocess_function = T2TPreprocessor(self.tuner.tokenizer)

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["train"]

        self.model = get_peft_model(self.model, self.tuner.config)
        print(self.model.print_trainable_parameters())

        optimizer = Optimizer(self.model, train_dataset, eval_dataset)
        optimizer.tune(self.tuner.tokenizer)


# Substitute value of x in above template
if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])

    trainer = Trainer()
    trainer.train()

    # Figure out how to push model to huggingface.
    peft_model_id = "opencui/test_PROMPT_TUNING_CAUSAL_LM"
    trainer.model.push_to_hub("opencui/test_PROMPT_TUNING_CAUSAL_LM", use_auth_token=True)

