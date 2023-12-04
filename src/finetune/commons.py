import re
from abc import ABC
import abc
from typing import Optional
from dataclasses import dataclass

import datasets
from datasets import Dataset
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode

from core.config import LugConfig
from core.annotation import Schema
from core.retriever import create_index, HybridRetriever
from finetune.embedding import create_sentence_pair_for_description, create_sentence_pair_for_exemplars


@dataclass
class AnnotatedExemplar:
    """
    expression examples, if the expected_slots is empty, this can be used for both skills and slots.
    """
    id: str
    utterance: str
    template: str
    owner: str
    arguments: Optional[dict] = None
    expectations: Optional[list] = None

    def flatten(self):
        return {
            "id": self.id,
            "utterance": self.utterance,
            "template": self.template,
            'owner': self.owner,
            "arguments": str(self.arguments),
            "expectations": str(self.expectations)
        }


def has_no_intent(label: str):
    return label == "NONE"


def build_nodes_from_dataset(module: str, dataset: Dataset, nodes):
    for item in dataset:
        utterance = item['template']
        label = item["owner"]
        if has_no_intent(label):
            continue
        nodes.append(
            TextNode(
                text=utterance,
                id_=item['id'],
                metadata={"arguments": item["arguments"], "owner": label, "module": module},
                excluded_embed_metadata_keys=["arguments", "owner", "module"]))


def build_dataset_index(tag: str, dsc: Dataset, output: str, embedding: BaseEmbedding):
    exemplar_nodes = []
    build_nodes_from_dataset(tag, dsc, exemplar_nodes)
    print(f"There are {len(exemplar_nodes)} exemplars.")
    create_index(output, "exemplar", exemplar_nodes, embedding)


def create_full_exemplar(id, utterance, intent, slots, spans, expectations=[]) -> AnnotatedExemplar:
    '''
    replacing the slot val with the slot name,to avoid match the short slot val which may be included in other
    long slot val, we need sort by the length of the slot val
    '''
    if not spans:
        return AnnotatedExemplar(id, utterance, utterance, intent, slots, expectations)
    single_dict = dict()

    for key, values in slots.items():
        for value in values:
            single_dict[value] = key

    spans = sorted(spans, key=lambda x: x[0])
    res_utterance = utterance[:spans[0][0]]
    for i, (cur_start, cur_end) in enumerate(spans):
        # if len(string_list) >=2:
        #     print("sub string",utterance[cur_start:cur_end])
        res_utterance = res_utterance + ' < ' + single_dict[utterance[cur_start:cur_end]] + ' > '
        if i == len(spans) - 1:
            res_utterance = res_utterance + utterance[cur_end:]
        else:
            res_utterance = res_utterance + utterance[cur_end:spans[i + 1][0]]
    return AnnotatedExemplar(id, utterance, res_utterance, intent, slots, expectations)


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
    def build(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


class ValueCollectable:
    @abc.abstractmethod
    def collect_slot_values(self, split):
        """This should collect all the values for all the slot"""
        return


class LoadFactory(DatasetFactory):
    def __init__(self, dsf: DatasetFactory, path: str):
        self.path = path
        self.tag = dsf.tag
        self.schema = dsf.schema

    def build(self, split: str = "train") -> Dataset:
        return datasets.load_from_disk(f"{self.path}/datasets/{self.tag}/{split}")


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
            exemplar_retriever=HybridRetriever(path, "exemplar", LugConfig.exemplar_retrieve_topk))


def generate_sentence_pairs(dataset_infos: list[DatasetCreatorWithIndex]) -> Dataset:
    generators = []
    for dataset_info in dataset_infos:
        dataset = dataset_info.creator.build("train")
        generators.extend(
            create_sentence_pair_for_description(
                dataset_info.creator.schema.skills,
                dataset,
                dataset_info.desc_retriever
            ))
        generators.extend(
            create_sentence_pair_for_exemplars(
                dataset,
                dataset_info.exemplar_retriever
            ))
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


if __name__ == "__main__":
    print(LugConfig.embedding_model)
