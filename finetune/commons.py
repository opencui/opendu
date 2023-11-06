import json
from abc import ABC
import abc
from typing import Any

from dataclasses import dataclass, field
from functools import reduce
from dataclasses_json import dataclass_json
from datasets import Dataset, concatenate_datasets
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode

from converter.lug_config import LugConfig
from core.annotation import ModuleSchema
from core.retriever import build_nodes_from_skills, create_index, HybridRetriever
from finetune.embedding import create_sentence_pair_for_description, create_sentence_pair_for_exemplars


@dataclass_json
@dataclass
class AnnotatedExemplar:
    """
    expression examples, if the expected_slots is empty, this can be used for both skills and slots.
    """
    id: str = field(metadata={"required": True})
    utterance: str = field(metadata={"required": True})
    template: str = field(metadata={"required": True})
    owner: str = field(metadata={"required": False})
    arguments: dict[str, Any] = field(metadata={"required": False})
    expectations: list[str] = field(metadata={"required": False}, default_factory=list)


def has_no_intent(label: str):
    return label == "NONE"


def build_nodes_from_dataset(dataset: Dataset):
    nodes = []
    for item in dataset:
        utterance = item['utterance']
        label = item["owner"]
        if has_no_intent(label): continue
        nodes.append(
            TextNode(
                text=utterance,
                id_=item['id'],
                metadata={"arguments": item["arguments"], "owner": label},
                excluded_embed_metadata_keys=["arguments", "owner"]))
    return nodes


def build_dataset_index(dsc: Dataset, output: str, embedding: BaseEmbedding):
    exemplar_nodes = build_nodes_from_dataset(dsc)
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
    domain: ModuleSchema

    @abc.abstractmethod
    def build(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


@dataclass
class DatasetsCreator(DatasetFactory):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dscs=list[DatasetFactory]):
        self.domain = ModuleSchema(
            skills=reduce(lambda x, y: {**x, **y}, [dsc.domain.skills for dsc in dscs]),
            slots=reduce(lambda x, y: {**x, **y}, [dsc.domain.arguments for dsc in dscs])
        )
        self.dscs = dscs

    def build(self, split):
        datasets = [dsc.build(split) for dsc in self.dscs]
        return concatenate_datasets(**datasets)


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
                dataset_info.creator.domain.skills,
                dataset,
                dataset_info.desc_retriever
            ))
        generators.extend(
            create_sentence_pair_for_exemplars(
                dataset,
                dataset_info.exemplar_retriever
            ))
    return generators


if __name__ == "__main__":
    print(LugConfig.embedding_model)
