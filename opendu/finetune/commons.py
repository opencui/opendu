import abc
import json
import re
import glob

from abc import ABC
from collections import defaultdict
from random import sample, seed
from datasets import Dataset, load_dataset, concatenate_datasets
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode


from opendu.core.annotation import Schema, MatchReplace, get_value
from opendu.core.config import RauConfig
from opendu.core.retriever import create_index
from opendu.finetune.structure_converter import FullExemplar, YniConverter
from opendu.finetune.prompt_converter import SkillBcPromptConverter

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


class RawJsonDatasetFactory(SchemaDatasetFactory, ABC):
    def __init__(self, path, tag=None, prefix=""):
        self.path = path
        schema_dict = json.load(open(f"{path}/schema.json"))
        self.schema = Schema(**schema_dict)
        self.files = {
            "train": f"{self.path}/{prefix}.jsonl",
            "validation": f"{self.path}/{prefix}.jsonl",
        }
        self.datasets = load_dataset('json', data_files=self.files)
        self.tag = tag

    def __getitem__(self, split: str = "train") -> Dataset:
        return self.datasets[split] if item in self.files else None


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

    def __getitem__(self, split: str) -> Dataset:
        return self.datasets[split]



class DatasetFactoryMerger(SchemaDatasetFactory, ABC):
    def __init__(self, factories):
        self.factories = factories

    def __getitem__(self, split: str) -> Dataset:
        datasets = []
        for factory in self.factories:
            datasets.append(factory[split])
        return concatenate_datasets(datasets).shuffle(seed=42)


# This inference is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
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


class PromptedFactory(DatasetFactory):
    __metaclass__ = abc.ABCMeta
    def __init__(self, file, unused_columns):
        self.file = file
        self.converters = [SkillBcPromptConverter()]
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
        dataset = load_dataset('json', data_files=self.file)[split]
        return dataset.map(self.convert_one, batched=True, remove_columns=self.columns)


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