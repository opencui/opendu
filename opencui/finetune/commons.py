import abc
import json
import random
import re
import glob

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from random import sample, seed
from typing import Optional

from dataclasses_json import dataclass_json
from datasets import Dataset, load_dataset, concatenate_datasets
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode

from opencui.core.prompt import (PybarsPrompt, MulticlassSkillPrompts, BinarySkillPrompts,
                                 ExemplarPrompts, DescriptionPrompts, BoolPrompts, YniPrompts, ExtractiveSlotPrompts)
from opencui.core.annotation import Schema, Exemplar, ListRecognizer, OwnerMode, ExactMatcher, MatchReplace, get_value
from opencui.core.config import RauConfig
from opencui.core.retriever import create_index, ContextRetriever
from opencui.finetune.phase1_converter import AnnotatedExemplar, TrainPhase1Converter
from opencui.finetune.phase2_converter import LabeledMatchingData, PromptConverter


def build_nodes_from_dataset(module: str, dataset: Dataset, nodes):
    pattern = re.compile(r"<(.+?)>")
    for item in dataset:
        arguments = json.loads(item["arguments"].replace("\'", "\""))
        utterance = item["utterance"]
        # Do not trust the original template.
        template = AnnotatedExemplar.extract_template(utterance, arguments)

        if template is None:
            template = item["template"]

        text = pattern.sub(MatchReplace(lambda x: x.replace("_", " ")), template)
        nodes.append(
            TextNode(
                text=text,
                id_=item["id"],
                metadata={
                    "arguments": item["arguments"],
                    "template": template,
                    "owner": (item["owner"]),
                    "owner_mode": item["owner_mode"],
                    "context_frame": get_value(item, "context_frame", ""),
                    "context_slot": get_value(item, "context_slot", ""),
                    "module": module,
                },
                excluded_embed_metadata_keys=[
                    "arguments", "owner", "module", "owner_mode", "context_frame", "context_slot", "template"],
            )
        )


def build_dataset_index(tag: str, dsc: Dataset, output: str, embedding: BaseEmbedding):
    exemplar_nodes = []
    build_nodes_from_dataset(tag, dsc, exemplar_nodes)
    print(f"There are {len(exemplar_nodes)} exemplars.")
    create_index(output, "exemplar", exemplar_nodes, embedding)


@dataclass
class DatasetFactory(ABC):
    __metaclass__ = abc.ABCMeta
    tag: str

    @abc.abstractmethod
    def __getitem__(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


#
# This is need to create the different dataset based on prompt templating.
# We expect the input dataset has utterance field.
# We need to make sure the output dataset has input/output field,
@dataclass
class SchemaDatasetFactory(DatasetFactory, ABC):
    __metaclass__ = abc.ABCMeta
    schema: Schema

@dataclass
class MappedDatasetDict(ABC):
    def __init__(self, ds_dict, train="train", validation="validation"):
        self.dict = ds_dict
        self.train = train
        self.validation = validation

    def __getitem__(self, split):
        if split == "train":
            return self.dict[self.train]
        if split == "validation":
            return self.dict[self.validation]
        return self.dict[split]


def collect_slot_values(dataset):
    entities = {}
    for exemplar in dataset:
        slot_values = eval(exemplar["arguments"])
        for key, values in slot_values.items():
            if key not in entities.keys():
                entities[key] = set()
            for value in values:
                entities[key].add(value)
    return entities


class JsonDatasetFactory(SchemaDatasetFactory, ABC):
    def __init__(self, path, tag=None, prefix=""):
        self.path = path
        schema_dict = json.load(open(f"{path}/schema.json"))
        self.schema = Schema.from_dict(schema_dict)
        files = {
            "train": f"{self.path}/{prefix}train.jsonl",
            "test": f"{self.path}/{prefix}test.jsonl",
            "validation": f"{self.path}/{prefix}validation.jsonl",
        }
        self.datasets = load_dataset('json', data_files=files)
        self.tag = tag

    def __getitem__(self, item):
        return self.datasets[item]


class JsonBareDatasetFactory(SchemaDatasetFactory, ABC):
    def __init__(self, path, tag=None, prefix=""):
        self.path = path
        files = {
            "train": f"{self.path}/{prefix}train.jsonl",
            "test": f"{self.path}/{prefix}test.jsonl",
            "validation": f"{self.path}/{prefix}validation.jsonl",
        }
        self.datasets = load_dataset('json', data_files=files)
        self.tag = tag

    def __getitem__(self, item):
        return self.datasets[item]



class DatasetFactoryMerger(SchemaDatasetFactory, ABC):
    def __init__(self, factories):
        self.factories = factories

    def __getitem__(self, split):
        datasets = []
        for factory in self.factories:
            datasets.append(factory[split])
        return concatenate_datasets(datasets).shuffle(seed=42)


# This inference is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
@dataclass
class ConvertedFactory(DatasetFactory):
    __metaclass__ = abc.ABCMeta
    def __init__(
        self,
        dsf: DatasetFactory,
        converters,
        unused_columns,
    ):
        self.creator = dsf
        self.converters = converters
        self.columns = unused_columns

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
class PromptedFactory(DatasetFactory):
    __metaclass__ = abc.ABCMeta
    def __init__(self, file, unused_columns):
        self.file = file
        self.converters = [PromptConverter()]
        self.columns = unused_columns
        self.tag = file.split("/")[3]

    def convert_one(self, item):
        ins = []
        outs = []
        for convert in self.converters:
            convert(item, ins, outs)
        assert len(ins) == len(outs)
        return {"input": ins, "output": outs}

    def __getitem__(self, split: str) -> Dataset:
        if split != "train": return None
        dataset = load_dataset('json', data_files=self.file)[split]
        return dataset.map(self.convert_one, batched=True, remove_columns=self.columns)


# Here we create the dataset factory for skills
def load_skill_factory(skill_modes, factories):
    # make sure run build_skill_dataset first.
    for skill_mode in skill_modes:
        factories.append(
            JsonDatasetFactory("./dugsets/sgd/", "sgd", f"{skill_mode}-{RauConfig.get().skill_prompt}.")
        )

def load_extractive_slot_factory(converted_factories):
    converted_factories.append(
        DatasetFactoryMerger([
            JsonBareDatasetFactory("./dugsets/sgd/", "sgd", "slots-"),
            JsonBareDatasetFactory("./dugsets/conll03/", "ner"),
        ])
    )

def load_nli_factory(converted_factories):
    # Here we assume the raw input is sentence, focus and label (positive, negative and neutral)
    converter = YniConverter(YniPrompts[RauConfig.get().yni_prompt])
    columns = ["question", "response", "label"]
    converted_factories.append(
        ConvertedFactory(JsonBareDatasetFactory("./dugsets/yni/", "yni"), [converter], columns)
    )


def load_bot_factory(converted_factories):
    # this is used to extract all the datasets from labeling process and make it available.
    # We assume that there are botsets/{lang}/{bots}/
    matching_data_directories = glob.glob("./botsets/en/*/MatchLabeledData.json")
    columns = ['_created_at', '_id', 'bot', 'context', 'decision', 'lang', 'matchType', 'owner', 'reference', 'userId', 'userOrg', 'utterance']
    for directory in matching_data_directories:
        # Add prompt to it.
        converted_factories.append(PromptedFactory(directory, columns))


# Load training set, based on what is inside the --training_mode desc-exemplar-extractive-slot
def load_training_dataset(args):
    converted_factories = []
    load_bot_factory(converted_factories)
    if "desc" in args.training_mode:
        print("load desc dataset")
        load_skill_factory(["desc"], converted_factories)
    if "exemplar" in args.training_mode:
        print("load exemplar dataset")
        load_skill_factory(["exemplar"], converted_factories)
    if "extractive_slot" in args.training_mode:
        print("load slot dataset")
        load_extractive_slot_factory(converted_factories)
    if "nli" in args.training_mode:
        print("load nli dataset")
        load_nli_factory(converted_factories)

    assert len(converted_factories) != 0

    # If we debug dataset, we do not train.
    if args.debug_dataset:
        count = 0
        for factory in converted_factories:
            ds = factory["train"]
            for item in ds:
                print(json.dumps(item, indent=2))
                count += 1
        print(count)
        exit(0)
    return converted_factories


def print_factories(factories):
    for factory in factories:
        ds = factory.__getitem__("train")
        count = 0
        for item in ds:
            print(item)
            count += 1
        print(f"There are {count} instances")


def purge_dataset(dataset, k=32, extract=lambda x: x["tag"]):
    # make it somewhat repeatable
    seed(42)

    def uptok(items):
        if len(items) < k:
            return items
        else:
            return sample(items, k=k)

    intents = defaultdict(list)
    utterances = set()
    count = 0
    for item in dataset:
        utterance = item["utterance"].lower()
        if utterance not in utterances:
            utterances.add(utterance)
            intents[extract(item)].append(item)
        else:
            count += 1
    print(f"There are {len(intents)} intents: {intents.keys()} with {count} duplicates.")
    return [example for examples in intents.values() for example in uptok(examples)]


class SlotSimplifier:
    def __init__(self):
        self.pattern = re.compile(r'\[(.*?)\]')

    def __call__(self, text):
        return re.findall(self.pattern, text)


class SlotFinalizer:
    def __init__(self):
        self.extract = SlotSimplifier()
        self.prefix = ["person", "|"]
    def __call__(self, text):
        matches = self.extract(text)
        payloads = list(map(lambda x: x.split("|")[0].strip(), matches))
        return " ".join(payloads)

if __name__ == "__main__":
    factories = []
    load_bot_factory(factories)
    factory = factories[0]
    print(factory)
    trainset = factory["train"]
    for item in trainset:
        print(item)