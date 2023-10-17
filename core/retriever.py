#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import logging
from datasets import Dataset
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode, NodeWithScore
from llama_index import QueryBundle
from builders.viggo import Viggo
from core.commons import DatasetCreator
from core.embedding import get_embedding

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# This is used to create the retriever so that we can get dynamic exemplars into understanding.
def create_index(path: str, dataset_creators: list[DatasetCreator]):
    nodes = []
    for creator in dataset_creators:
        dataset = creator.build("train")
        for item in dataset:
            utterance = item['utterance']
            nodes.append(
                TextNode(
                    text=utterance,
                    id_=item['id'],
                    metadata={"target_full": item["target_full"], "target_intent": item["target_intent"]},
                    excluded_embed_metadata_keys=["target_full", "target_intent"]))

    # Init download hugging fact model
    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=get_embedding(),
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


#
# There are four kinds of mode: embedding, keyword, AND and OR.
#
class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(self, path: str, topk: int = 8, mode: str = "embedding") -> None:
        """Init params."""
        if mode not in ("embedding", "keyword", "AND", "OR"):
            raise ValueError("Invalid mode.")

        embedding = get_embedding()
        service_context = ServiceContext.from_defaults(
            llm=None,
            llm_predictor=None,
            embed_model=embedding)
        storage_context = StorageContext.from_defaults(persist_dir=path)
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


if __name__ == "__main__":
    output = "./index/viggo/"
    create_index(output, [Viggo()])

    retriever = HybridRetriever(output)
    utterance = "Are you into third person PC games like Little Big Adventure?"
    results = retriever.retrieve(utterance)
    for result in results:
        print(result.node)
        print("\n")
