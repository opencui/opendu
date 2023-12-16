import abc
import json
import random
import re

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from random import sample, seed
from typing import Optional

from dataclasses_json import dataclass_json
from datasets import Dataset, load_dataset
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode

from opencui import Prompt, MulticlassSkillPrompts, BinarySkillPrompts, LayeredPrompts
from opencui.core.annotation import Schema, Exemplar, ListRecognizer
from opencui.core.config import LugConfig
from opencui.core.retriever import HybridRetriever, create_index, ContextRetriever
from opencui.finetune.embedding import (
    create_sentence_pair_for_description, create_sentence_pair_for_exemplars)


@dataclass_json
@dataclass
class AnnotatedExemplar:
    """
    expression examples, if the expected_slots is empty, this can be used for both skills and slots.
    """

    id: str
    owner: str
    utterance: str
    arguments: dict
    owner_mode: str = "normal"
    template: str = None
    expectations: Optional[list] = None

    def flatten(self):
        return {
            "id": self.id,
            "owner": self.owner,
            "utterance": self.utterance,
            "arguments": str(self.arguments),
            "owner_mode": str(self.extended),
            "template": self.template,
            "expectations": str(self.expectations),
        }

    @staticmethod
    def get_span(word, sentence):
        # Construct a regular expression pattern to find the word with boundaries and punctuation
        pattern = r'\b' + re.escape(word) + r'\b'

        # Search for the pattern in the sentence
        return re.findall(pattern, sentence), re.search(pattern, sentence)

    @staticmethod
    def extract_template(utterance, arguments):
        if len(arguments) == 0:
            return utterance

        single_dict = dict()
        spans = []
        for key, values in arguments.items():
            for value in values:
                single_dict[value] = key
                found, match = AnnotatedExemplar.get_span(value, utterance)
                if len(found) != 1:
                    return None
                spans.append(match.span())

        spans = sorted(spans, key=lambda x: x[0])
        res_utterance = utterance[: spans[0][0]]
        for i, (cur_start, cur_end) in enumerate(spans):
            # if len(string_list) >=2:
            #     print("sub string",utterance[cur_start:cur_end])
            res_utterance = (
                    res_utterance + " < " + single_dict[utterance[cur_start:cur_end]] + " > "
            )
            if i == len(spans) - 1:
                res_utterance = res_utterance + utterance[cur_end:]
            else:
                res_utterance = res_utterance + utterance[cur_end: spans[i + 1][0]]
        return res_utterance


def has_no_intent(label: str):
    return label == "NONE"


def build_nodes_from_dataset(module: str, dataset: Dataset, nodes):
    for item in dataset:
        print(item["arguments"])
        arguments = json.loads(item["arguments"].replace("\'", "\""))
        utterance = item["utterance"]
        template = AnnotatedExemplar.extract_template(utterance, arguments)

        print(template)
        if template is None:
            utterance = item["template"]
        else:
            utterance = template

        label = item["owner"]
        if has_no_intent(label):
            continue
        nodes.append(
            TextNode(
                text=utterance,
                id_=item["id"],
                metadata={
                    "arguments": item["arguments"],
                    "owner": label,
                    "module": module,
                },
                excluded_embed_metadata_keys=["arguments", "owner", "module"],
            )
        )


def build_dataset_index(tag: str, dsc: Dataset, output: str, embedding: BaseEmbedding):
    exemplar_nodes = []
    build_nodes_from_dataset(tag, dsc, exemplar_nodes)
    print(f"There are {len(exemplar_nodes)} exemplars.")
    create_index(output, "exemplar", exemplar_nodes, embedding)


def create_full_exemplar(
        id, utterance, intent, slots, spans, expectations=[]
) -> AnnotatedExemplar:
    """
    replacing the slot val with the slot name,to avoid match the short slot val which may be included in other
    long slot val, we need sort by the length of the slot val
    """
    if not spans:
        return AnnotatedExemplar(id, intent, utterance, utterance, slots, expectations)
    single_dict = dict()

    for key, values in slots.items():
        for value in values:
            single_dict[value] = key

    spans = sorted(spans, key=lambda x: x[0])
    res_utterance = utterance[: spans[0][0]]
    for i, (cur_start, cur_end) in enumerate(spans):
        # if len(string_list) >=2:
        #     print("sub string",utterance[cur_start:cur_end])
        res_utterance = (
                res_utterance + " < " + single_dict[utterance[cur_start:cur_end]] + " > "
        )
        if i == len(spans) - 1:
            res_utterance = res_utterance + utterance[cur_end:]
        else:
            res_utterance = res_utterance + utterance[cur_end: spans[i + 1][0]]
    return AnnotatedExemplar(id, intent, utterance, res_utterance, slots, expectations)


#
# This is need to create the different dataset based on prompt templating.
# We expect the input dataset has utterance field.
# We need to make sure the output dataset has input/output field,
@dataclass
class DatasetFactory(ABC):
    __metaclass__ = abc.ABCMeta
    tag: str
    schema: Schema

    @abc.abstractmethod
    def __getitem__(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


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


@dataclass
class DatasetCreatorWithIndex:
    creator: DatasetFactory
    desc_retriever: HybridRetriever
    exemplar_retriever: HybridRetriever

    @classmethod
    def build(cls, creator: DatasetFactory, path: str):
        return DatasetCreatorWithIndex(
            creator=creator,
            desc_retriever=HybridRetriever(path, "desc", LugConfig.desc_retrieve_topk),
            exemplar_retriever=HybridRetriever(
                path, "exemplar", LugConfig.exemplar_retrieve_topk
            ),
        )


def generate_sentence_pairs(dataset_infos: list[DatasetCreatorWithIndex]) -> Dataset:
    generators = []
    for dataset_info in dataset_infos:
        dataset = dataset_info.creator["train"]
        generators.extend(
            create_sentence_pair_for_description(
                dataset_info.creator.schema.skills, dataset, dataset_info.desc_retriever
            )
        )
        generators.extend(
            create_sentence_pair_for_exemplars(dataset, dataset_info.exemplar_retriever)
        )
    return generators


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


# Some time, the original data are over sampled, we need to purge the extra things.
def purge_dataset(dataset, k=32):
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
            intents[item["owner"]].append(item)
        else:
            count += 1
    print(f"There are {len(intents)} intents: {intents.keys()} with {count} duplicates.")
    return [example for examples in intents.values() for example in uptok(examples)]


class JsonDatasetFactory(DatasetFactory, ABC):
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

    def extra_tokens(self):
        return json.load(open(f"{self.path}/extra.tokens"))

    def __getitem__(self, item):
        return self.datasets[item]


# This inference is responsible for convert the exemplars in the original dataset into what is needed
# by generation fine-tuning. The assumed the columns are input and output, and we added id for debugging
# purpose.
class TrainConverter(ABC):
    prompt: Prompt

    @abc.abstractmethod
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
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]
            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text)
                for node in nodes
            ]
            owner = batch["owner"][idx]

            # How can we reduce the need for

            neg_owners = [
                node.metadata["owner"]
                for node in nodes
                if node.metadata["owner"] != owner
            ]

            # randomly filter one neg skills and exemplars
            if len(neg_owners) != 0:
                neg_owner = random.choice(neg_owners)
                rm_neg_exemplars = [
                    exemplar for exemplar in exemplars if exemplar.owner != neg_owner
                ]
                rm_neg_skills = [
                    skill for skill in skills if skill["name"] != neg_owner
                ]

                # Without exemplars.
                random.shuffle(rm_neg_skills)
                input_dict = {
                    "utterance": utterance,
                    "examples": [],
                    "skills": rm_neg_skills,
                }
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner)}</s>")

                # With exemplars.
                if len(rm_neg_exemplars) != 0:
                    random.shuffle(rm_neg_exemplars)
                    input_dict = {
                        "utterance": utterance,
                        "examples": rm_neg_exemplars,
                        "skills": rm_neg_skills,
                    }
                    ins.append(self.prompt(input_dict))
                    outs.append(f"{json.dumps(owner)}</s>")

            # Try to filter the pos skills and exemplars
            rm_pos_exemplars = [
                exemplar for exemplar in exemplars if exemplar.owner != owner
            ]
            rm_pos_skills = [skill for skill in skills if skill["name"] != owner]

            random.shuffle(rm_pos_skills)
            input_dict = {
                "utterance": utterance,
                "examples": [],
                "skills": rm_pos_skills,
            }
            ins.append(self.prompt(input_dict))
            outs.append(f"{json.dumps(None)}</s>")

            if len(rm_pos_exemplars) != 0:
                random.shuffle(rm_pos_exemplars)
                input_dict = {
                    "utterance": utterance,
                    "examples": rm_pos_exemplars,
                    "skills": rm_pos_skills,
                }
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
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]
            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text)
                for node in nodes
            ]
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
                    {
                        "template": exemplar.template,
                        "target": target,
                        "decision": target == exemplar.owner,
                    }
                    for exemplar in exemplars
                ]

                input_dict = {
                    "utterance": utterance,
                    "examples": exemplar_dicts,
                    "skill": skill_map[target],
                }
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
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]
            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text)
                for node in nodes
            ]
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
            pattern = re.compile("|".join(map(re.escape, strings_to_check)))
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
                values = set(
                    ListRecognizer.find_matches(self.patterns, slot_name, utterance)
                )
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
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                    # then with values.
                    input_dict["values"] = values
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
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


# This inference is needed for cases where users' utterance is response to bot's prompt questions, and
# needs the abstractive understanding instead of extractive understanding.
# This is needed to determine the intention, intended function or skill
# class BooleanConverter
@dataclass
class PromptedFactory(DatasetFactory):
    __metaclass__ = abc.ABCMeta
    skill_columns = [
        "id",
        "utterance",
        "template",
        "owner",
        "extended",
        "arguments",
        "expectations",
    ]

    def __init__(
        self,
        dsf: DatasetFactory,
        convert: list[TrainConverter],
        unused_columns=skill_columns,
    ):
        self.creator = dsf
        self.converters: list[TrainConverter] = convert
        self.columns = unused_columns

    def extra_tokens(self):
        return list(
            set(
                [
                    token
                    for converter in self.converters
                    for token in converter.prompt.extra_tokens
                ]
            )
        )

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
