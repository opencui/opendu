import json
from abc import ABC
import abc
from dataclasses import dataclass, field
from functools import reduce
from dataclasses_json import dataclass_json
from datasets import Dataset, concatenate_datasets

from converter.lugconfig import LugConfig
from core.annotation import ModuleSpec
from core.embedding import EmbeddingStore
from core.prompt import Prompt
from core.retriever import build_nodes_from_skills, build_nodes_from_dataset, create_index, HybridRetriever


@dataclass
@dataclass_json
class Expression:
    """
    expression examples
    """
    id: str = field(metadata={"required": True})
    utterance: str = field(metadata={"required": True})
    target_intent: str = field(metadata={"required": True})
    target_slots: dict[str, str] = field(metadata={"required": True})
    spans: list[tuple[int, int]] = field(metadata={"required": True})
    exemplar: str = field(metadata={"required": True})

    def __init__(self, id, utterance, intent, slots, spans):
        self.id = id
        self.utterance = utterance
        self.target_intent = intent
        self.target_slots = slots  # dict to store slot, value pairs
        self.spans = spans
        self.exemplar = Expression.generate_expression_template(utterance, slots, spans)

    @classmethod
    def generate_expression_template(cls, utterance, slot_dict, spans):
        '''
        replacing the slot val with the slot name,to avoid match the short slot val which may be included in other
        long slot val, we need sort by the length of the slot val
        '''
        if spans == []:
            return utterance
        single_dict = dict()

        for key, values in slot_dict.items():
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

        return res_utterance


#
# This is need to create the different dataset based on prompt templating.
# We expect the input dataset has utterance field.
# We need to make sure the output dataset has input/output field,
@dataclass
class DatasetFactory(ABC):
    __metaclass__ = abc.ABCMeta
    domain: ModuleSpec

    @abc.abstractmethod
    def build(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


@dataclass
class DatasetsCreator(DatasetFactory):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dscs=list[DatasetFactory]):
        self.domain = ModuleSpec(
            skills=reduce(lambda x, y: {**x, **y}, [dsc.domain.skills for dsc in dscs]),
            slots=reduce(lambda x, y: {**x, **y}, [dsc.domain.target_slots for dsc in dscs])
        )
        self.dscs = dscs

    def build(self, split):
        datasets = [dsc.build(split) for dsc in self.dscs]
        return concatenate_datasets(**datasets)


@dataclass
class DatasetFactoryWrapper(DatasetFactory):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dsf: DatasetFactory, prompt: Prompt):
        self.domain = dsf.domain
        self.prompt = prompt

    def build(self, split: str) -> Dataset:
        dataset = self.creator.build(split)
        return dataset.map(lambda x: {"input": self.prompt(x)})

    @classmethod
    def build_index(cls, dsc: DatasetFactory, output: str = "./output/"):
        desc_nodes = build_nodes_from_skills(dsc.domain.skills)
        exemplar_nodes = build_nodes_from_dataset(dsc.build("train"))

        create_index(output, "desc", desc_nodes, EmbeddingStore.for_description())
        create_index(output, "exemplars", exemplar_nodes, EmbeddingStore.for_exemplar())



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


if __name__ == "__main__":
    print(LugConfig.embedding_model)

