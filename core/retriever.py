#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import logging
from dataclasses import dataclass

from factories import Dataset
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode, NodeWithScore
from llama_index import QueryBundle
from core.commons import DatasetFactory, SkillInfo, Config
from core.embedding import EmbeddingStore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def has_no_intent(label: str):
    return label in {"NONE"}


def build_nodes_from_dataset(dataset: Dataset):
    nodes = []
    for item in dataset:
        utterance = item['exemplar']
        label = item["target_intent"]
        if has_no_intent(label):
            nodes.append(
                TextNode(
                    text=utterance,
                    id_=item['id'],
                    metadata={"target_slots": item["target_slots"], "target_intent": label},
                    excluded_embed_metadata_keys=["target_slots", "target_intent"]))
    return nodes


def build_nodes_from_skills(skills: dict[str, SkillInfo]):
    nodes = []
    for label, skill in skills.items():
        desc = skill["description"]
        name = skill["name"]
        nodes.append(
            TextNode(
                text=desc,
                id_=label,
                metadata={"target_intent": name},
                excluded_embed_metadata_keys=["target_intent"]))
    return nodes


# This is used to create the retriever so that we can get dynamic exemplars into understanding.
def create_index(base: str, tag: str, nodes: list[TextNode]):
    path = f"{base}/{tag}/"
    embedding: BaseEmbedding = EmbeddingStore.get_embedding_by_task(tag)
    # Init download hugging fact model
    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=embedding,
    )

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    try:
        embedding_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            service_context=service_context)

        keyword_index = SimpleKeywordTableIndex(
            nodes,
            storage_context=storage_context,
            service_context=service_context)

        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=path)
        keyword_index.set_index_id("keyword")
        keyword_index.storage_context.persist(persist_dir=path)
    except Exception as e:
        print(str(e))
        shutil.rmtree(path, ignore_errors=True)


def build_exemplar_index(dsc: DatasetFactory, output: str):
    exemplar_nodes = build_nodes_from_dataset(dsc.build("train"))
    create_index(output, "exemplar", exemplar_nodes)


def build_desc_index(dsc: DatasetFactory, output: str):
    desc_nodes = build_nodes_from_skills(dsc.domain.skills)
    create_index(output, "desc", desc_nodes)


def get_retrievers(path: str):
    return [
        HybridRetriever(path, "desc", Config.desc_retrieve_topk),
        HybridRetriever(path, "exemplar", Config.exemplar_retrieve_topk)]


#
# There are four kinds of mode: embedding, keyword, AND and OR.
#
class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(self, path: str, tag: str, topk: int = 8, mode: str = "embedding") -> None:
        """Init params."""
        if mode not in ("embedding", "keyword", "AND", "OR"):
            raise ValueError("Invalid mode.")

        embedding = EmbeddingStore.get_embedding_by_task(tag)
        service_context = ServiceContext.from_defaults(
            llm=None,
            llm_predictor=None,
            embed_model=embedding)
        storage_context = StorageContext.from_defaults(persist_dir=f"{path}/{tag}/")
        embedding_index = load_index_from_storage(
            storage_context,
            index_id="embedding",
            service_context=service_context)
        keyword_index = load_index_from_storage(
            storage_context,
            index_id="keyword",
            service_context=service_context)

        self._vector_retriever = VectorIndexRetriever(
            index=embedding_index,
            similarity_top_k=topk)
        self._keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "embedding":
            retrieve_ids = vector_ids
        elif self._mode == "keyword":
            retrieve_ids = keyword_ids
        elif self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


@dataclass
class DatasetCreatorWithIndex:
    creator: DatasetFactory
    desc_retriever: HybridRetriever
    exemplar_retriever: HybridRetriever

    @classmethod
    def build(cls, creator: DatasetFactory, path: str):
        return DatasetCreatorWithIndex(
            creator=creator,
            desc_retriever=HybridRetriever(path, "desc", Config.desc_retrieve_topk),
            exemplar_retriever=HybridRetriever(path, "exemplar", Config.exemplar_retrieve_topk))


def compute_k(dataset: Dataset, output: str, tag: str, topk: int = 3):
    retriever = HybridRetriever(output, tag, topk=8)
    counts = [0, 0]
    for item in dataset:
        nodes = retriever.retrieve(item["utterance"])
        intents = set()
        lintents = []
        for result in nodes:
            intent = result.node.metadata["target_intent"]
            if intent not in intents:
                intents.add(intent)
                lintents.append(intent)
            if len(lintents) >= topk:
                break
        counts[0] += 1
        if item["target_intent"] in lintents[0:topk]:
            counts[1] += 1

    return counts


def compute_hits(dataset: Dataset, output: str, topk: int):
    retriever = HybridRetriever(output, "desc", topk=topk)
    counts = [0, 0]
    for item in dataset:
        nodes = retriever.retrieve(item["utterance"])
        intents = {result.node.metadata["target_intent"] for result in nodes}
        counts[0] += 1
        name = item["target_intent"]
        if name in intents or name == "NONE":
            counts[1] += 1
        else:
            print(f'{name}:{item["utterance"]} not in {intents}')

    return counts


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    output = "./index/sgdskill/"
    from factories.sgd import SGDSkills
    dsc = SGDSkills("/home/sean/src/dstc8-schema-guided-dialogue/")

    Config.embedding_device = "cuda"
    dataset = dsc.build("train")
    print(compute_hits(dataset, output, 8))

    #print(compute_k(dataset, output, "exemplar"))
