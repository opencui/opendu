#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import shutil
from collections import defaultdict

from llama_index import (QueryBundle, ServiceContext, StorageContext, VectorStoreIndex, load_index_from_storage)
from llama_index.embeddings.base import BaseEmbedding
# Retrievers
from llama_index.retrievers import (BaseRetriever, BM25Retriever, VectorIndexRetriever)

from llama_index.schema import NodeWithScore, TextNode

from opencui.core.annotation import (FrameId, FrameSchema, Schema)
from opencui.core.config import LugConfig
from opencui.core.embedding import EmbeddingStore


def build_nodes_from_skills(module: str, skills: dict[str, FrameSchema], nodes):
    for label, skill in skills.items():
        desc = skill["description"]
        name = skill["name"]
        if desc.strip() == "":
            continue
        nodes.append(
            TextNode(
                text=desc,
                id_=label,
                metadata={
                    "owner": name,
                    "module": module,
                    "owner_mode": "normal"
                },
                excluded_embed_metadata_keys=["owner", "module", "owner_mode"],
            ))


# This is used to create the retriever so that we can get dynamic exemplars into understanding.
def create_index(base: str, tag: str, nodes: list[TextNode],
                 embedding: BaseEmbedding):
    path = f"{base}/{tag}/"
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
        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=path)
    except Exception as e:
        print(str(e))
        shutil.rmtree(path, ignore_errors=True)


def build_desc_index(module: str, dsc: Schema, output: str,
                     embedding: BaseEmbedding):
    desc_nodes = []
    build_nodes_from_skills(module, dsc.skills, desc_nodes)
    create_index(output, "desc", desc_nodes, embedding)


# This merge the result.
def merge_nodes(nodes0: list[NodeWithScore], nodes1: list[NodeWithScore])-> list[NodeWithScore]:
    nodes = {}
    scores = {}
    for ns in nodes0 + nodes1:
        if ns.node.text in nodes:
            scores[ns.node.text] += ns.score
        else:
            nodes[ns.node.text] = ns.node
            scores[ns.node.text] = ns.score

    res = [NodeWithScore(node=nodes[nid], score=scores[nid]) for nid in nodes.keys()]
    return sorted(res, key=lambda x: x.score, reverse=True)


#
class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""
    @staticmethod
    def load_hybrid_retriever(path: str, tag: str, topk: int = 8) -> None:
        embedding = EmbeddingStore.get_embedding_by_task(tag)
        service_context = ServiceContext.from_defaults(
            llm=None,
            llm_predictor=None,
            embed_model=embedding)

        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{path}/{tag}/")

            embedding_index = load_index_from_storage(
                storage_context,
                index_id="embedding",
                service_context=service_context)

            vector_retriever = VectorIndexRetriever(
                index=embedding_index,
                similarity_top_k=topk)
        # For now, we do index everytime we restart the inference.
            keyword_retriever = BM25Retriever.from_defaults(
                docstore=embedding_index.docstore, similarity_top_k=topk)
            return HybridRetriever(vector_retriever, keyword_retriever)
        except (ZeroDivisionError, FileNotFoundError) as error:
            print(error)
            return None

    def __init__(self, vec_retriever, word_retriever):
        self._vector_retriever = vec_retriever
        self._keyword_retriever = word_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        return merge_nodes(vector_nodes, keyword_nodes)


def dedup_nodes(old_results: list[TextNode], with_mode, arity=1):
    new_results = []
    intents = defaultdict(int)
    for item in old_results:
        intent = f'{item.metadata["owner_mode"]}.{item.metadata["owner"]}' if with_mode else item.metadata["owner"]
        if intents[intent] < arity:
            intents[intent] += 1
            new_results.append(item)
    return new_results


# This allows us to use the same logic on both the inference and fine-tuning side.
# This is used to create the context for prompt needed for generate the solution for skills.
class ContextRetriever:
    def __init__(self, module: Schema, d_retrievers, e_retriever):
        self.module = module
        self.desc_retriever = d_retrievers
        self.exemplar_retriever = e_retriever
        self.arity = LugConfig.get().exemplar_retrieve_arity
        self.extended_mode = False

    def retrieve_by_desc(self, query):
        # The goal here is to find the combined descriptions and exemplars.
        return self.desc_retriever.retrieve(query)

    def retrieve_by_exemplar(self, query):
        return self.exemplar_retriever.retrieve(query)

    def __call__(self, query):
        # The goal here is to find the combined descriptions and exemplars.

        if self.desc_retriever is not None:
            desc_nodes = [
                item.node for item in self.desc_retriever.retrieve(query)
            ]
        else:
            desc_nodes = []

        exemplar_nodes = [
            item.node for item in self.exemplar_retriever.retrieve(query)
        ]

        exemplar_nodes = dedup_nodes(exemplar_nodes, True, self.arity)
        all_nodes = dedup_nodes(desc_nodes + exemplar_nodes, False, 1)

        owners = [
            FrameId(item.metadata["module"], item.metadata["owner"])
            for item in all_nodes
        ]

        # Need to remove the bad owner/func/skill/intent.
        skills = [self.module.get_skill(owner) for owner in owners]
        return skills, exemplar_nodes


def load_context_retrievers(module: Schema, path: str):

    return ContextRetriever(
        module,
        HybridRetriever.load_hybrid_retriever(path, "desc", LugConfig.get().desc_retrieve_topk),
        HybridRetriever.load_hybrid_retriever(path,"exemplar", LugConfig.get().exemplar_retrieve_topk),
    )