import abc
import json
import os
import re
import glob

from abc import ABC
from collections import defaultdict
from random import sample, seed

from datasets import Dataset, load_dataset, concatenate_datasets
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode


from opendu.core.annotation import Schema, MatchReplace, get_value
from opendu.core.retriever import create_index
from opendu.finetune.structure_converter import FullExemplar


#
# We will have more than one task need to be handled.
# We assume the following steps in prepare the fine tuning datasets.
# 1. Convert the original dataset into some common format so that we can handle different dataset, including create
#    the schema.
# 2. Decide what task we will support, some time, we can use the side tasks.
# 3. Decide what retrieval we need for each task.
# 4. Decide want prompting we need for that task.
# 5. Generate the datasets for fine-tuning, make sure the label is meaningful.
#
# In there, we will have 1-3 as the phrase #1, and 4 as phrase #2, so that we can change prompt without
# change phase #1.


def build_nodes_from_dataset(module: str, dataset: Dataset, nodes):
    pattern = re.compile(r"<(.+?)>")
    for item in dataset:
        arguments = json.loads(item["arguments"].replace("\'", "\""))
        utterance = item["utterance"]
        # Do not trust the original template.
        template = FullExemplar.extract_template(utterance=utterance, arguments=arguments)

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


class DatasetFactory(ABC):
    __metaclass__ = abc.ABCMeta
    tag: str

    @abc.abstractmethod
    def __getitem__(self, split: str) -> Dataset:
        """This return the domain meta needed."""


#
# This is need to create the different dataset based on prompt templating.
# We expect the input dataset has utterance field.
# We need to make sure the output dataset has input/output field,
class SchemaDatasetFactory(DatasetFactory, ABC):
    __metaclass__ = abc.ABCMeta
    schema: Schema


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
        self.schema = Schema(**schema_dict)
        files = {
            "train": f"{self.path}/{prefix}train.jsonl",
            "test": f"{self.path}/{prefix}test.jsonl",
            "validation": f"{self.path}/{prefix}validation.jsonl",
        }
        self.datasets = load_dataset('json', data_files=files)
        self.tag = tag

    def __getitem__(self, split: str = "train") -> Dataset:
        return self.datasets[split]


#
# FineTuneDataset can be readily used for finetuning, they share the same structure, and
# can be used with different prompt.
class FtDatasetFactory(SchemaDatasetFactory, ABC):
    def __init__(self, path, converters=[], columns=[]):
        # When the schema.json exist.
        schema_path = f"{path}/schema.json"
        self.schema = json.load(open(schema_path)) if os.path.exists(schema_path) else None
        self.converters = converters
        self.unused_columns = columns

        # We assume that {train|test|dev}*.jsonl under path, and schema.json
        json_files = glob.glob(os.path.join(path, "*.jsonl"), recursive=True)
        print(f"json_files = {json_files}")
        self.datasets = {}
        for prefix in ["train", "dev", "test"]:
            self.datasets[prefix] = self.load_dataset(json_files, prefix)

    def convert_one(self, item):
        ins = []
        outs = []
        for convert in self.converters:
            convert(item, ins, outs)
        assert len(ins) == len(outs)
        return {"input": ins, "output": outs}

    def load_dataset(self, json_files, prefix: str):
        datasets = []
        for json_file in json_files:
            filename = os.path.basename(json_file)
            if filename.startswith(prefix):
                # Check if the key starts with "train"
                dataset_dict = load_dataset('json', data_files=json_file)
                # by the default, the dataset is loaded under the key train.
                dataset = dataset_dict["train"]
                dataset = dataset.map(self.convert_one, batched=True, remove_columns=self.unused_columns)
                datasets.append(dataset)
            print(f"process {json_file} with {prefix} with {self.converters}")
        return concatenate_datasets(datasets) if len(datasets) != 0 else None

    def __getitem__(self, split: str) -> Dataset:
        return self.datasets[split]


# We use this to merge multiple factory into one.
class MergedDatasetFactory(SchemaDatasetFactory, ABC):
    def __init__(self, factories):
        self.factories = factories

    def __getitem__(self, split: str) -> Dataset:
        datasets = []
        for factory in self.factories:
            dataset = factory[split]
            if dataset != None:
                datasets.append(factory[split])
        return concatenate_datasets(datasets).shuffle(seed=42) if len(datasets) != 0 else None

    def get(self, split:str) -> Dataset:
        return self[split]

# This inference is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
class BatchConvertedFactory(DatasetFactory):
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