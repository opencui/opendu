#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import shutil
from collections import defaultdict
from typing import Callable, List, Optional, cast

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
# Retrievers
from llama_index.core.retrievers import (BaseRetriever, VectorIndexRetriever)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore, TextNode, BaseNode

from opencui.core.annotation import (FrameId, FrameSchema, Schema, CamelToSnake, get_value)
from opencui.core.config import RauConfig
from opencui.core import embedding


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
    Settings.llm = None
    Settings.llm_predictor = None
    Settings.embed_model = embedding


    storage_context = StorageContext.from_defaults()
    print(f"Add {len(nodes)} nodes to {tag}")
    storage_context.docstore.add_documents(nodes)

    try:
        embedding_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context)
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
        if ns.node.id_ in nodes:
            scores[ns.node.id_] += ns.score
        else:
            nodes[ns.node.id_] = ns.node
            scores[ns.node.id_] = ns.score

    res = [NodeWithScore(node=nodes[nid], score=scores[nid]) for nid in nodes.keys()]
    return sorted(res, key=lambda x: x.score, reverse=True)



class EmbeddingRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search."""
    @staticmethod
    def load_retriever(path: str, tag: str, topk: int = 8) -> None:
        Settings.llm = None
        Settings.llm_predictor = None
        Settings.embed_model=embedding.EmbeddingStore.get_embedding_by_task(tag)

        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{path}/{tag}/")

            embedding_index = load_index_from_storage(
                storage_context,
                index_id="embedding")

            vector_retriever = VectorIndexRetriever(
                index=embedding_index,
                similarity_top_k=topk)

            return EmbeddingRetriever(vector_retriever)
        except (ZeroDivisionError, FileNotFoundError) as error:
            print(error)
            return None

    def __init__(self, vec_retriever):
        self._vector_retriever = vec_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        return self._vector_retriever.retrieve(query_bundle)

#
class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""
    @staticmethod
    def load_retriever(path: str, tag: str, topk: int = 8) -> None:
        Settings.llm = None
        Settings.llm_predictor = None
        Settings.embed_model=embedding.EmbeddingStore.get_embedding_by_task(tag)

        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{path}/{tag}/")

            embedding_index = load_index_from_storage(
                storage_context,
                index_id="embedding")

            vector_retriever = VectorIndexRetriever(
                index=embedding_index,
                similarity_top_k=topk)

            # For exemplar, the embedding and keyword need to use different
            # The reason we use original template is to reduce the casual match
            # related to slot name, since the original template use slot_label.
            keywords_nodes = []
            raw_nodes = list(embedding_index.docstore.docs.values())
            for node in raw_nodes:
                keywords_nodes.append(
                    TextNode(
                        text=node.metadata["template"],
                        id_=node.id_,
                        metadata=node.metadata,
                        excluded_embed_metadata_keys=["owner", "template_without_slot", "context_frame", "context_slot", "owner_mode", "template"],
                    )
                )

            # For now, we do index everytime we restart the inference.
            keyword_retriever = BM25Retriever.from_defaults(
                nodes=keywords_nodes, similarity_top_k=topk)
            return HybridRetriever(vector_retriever, keyword_retriever)
        except (ZeroDivisionError, FileNotFoundError) as error:
            print(error)
            return None

    def __init__(self, vec_retriever, word_retriever):
        self._vector_retriever = vec_retriever
        self._keyword_retriever = word_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        if not query_bundle.query_str.startswith("<") or not query_bundle.query_str.endswith(">"):
            print("hybrid search")
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
            return merge_nodes(vector_nodes, keyword_nodes)
        else:
            print("key word only search")
            return self._keyword_retriever.retrieve(query_bundle)


def dedup_nodes(old_results: list[TextNode], with_mode, arity=1):
    new_results = []
    intents = defaultdict(int)
    for item in old_results:
        intent = f'{item.metadata["owner_mode"]}.{item.metadata["owner"]}' if with_mode else item.metadata["owner"]
        if intents[intent] < arity:
            intents[intent] += 1
            new_results.append(item)
    return new_results


class ContextMatcher:
    def __init__(self, frame):
        self.frame = frame["frame"]
        self.slot = frame["slot"]

    def __call__(self, node: NodeWithScore):
        meta = node.node.metadata
        context_frame = get_value(meta, "context_frame")
        # make sure we have the slot, current assume the template use no space between slot and <>
        if node.node.text.find(f"<{self.slot}>") == -1:
            return False

        if context_frame is None or context_frame == "":
            return True
        else:
            return self.frame == context_frame


# This allows us to use the same logic on both the inference and fine-tuning side.
# This is used to create the context for prompt needed for generate the solution for skills.
class ContextRetriever:
    def __init__(self, module: Schema, d_retrievers, e_retriever):
        self.module = module
        self.desc_retriever = d_retrievers
        self.exemplar_retriever = e_retriever
        assert(e_retriever is not None)
        self.arity = RauConfig.get().exemplar_retrieve_arity
        self.extended_mode = False

    def retrieve_by_desc(self, query):
        # The goal here is to find the combined descriptions and exemplars.
        return self.desc_retriever.retrieve(query)

    def retrieve_by_exemplar(self, query):
        return self.exemplar_retriever.retrieve(query)

    def retrieve_by_expectation(self, expectations):
        # What if the query is too short, we use slot from expectation to help.
        # Note, this only works if text in node contains <label> not just slot name.
        # For now, this does not have an effect, because there is no <label> in the exemplar.
        slot_nodes = []
        for frame in expectations:
            slot = get_value(frame, 'slot')
            if slot is None or slot == "":
                continue

            query = f"<{slot}>"
            match = ContextMatcher(frame)
            nodes = self.exemplar_retriever.retrieve(query)
            # make sure the frame also match
            slot_nodes.extend(filter(match, nodes))
        return slot_nodes

    def __call__(self, query):
        # The goal here is to find the combined descriptions and exemplars.
        if self.desc_retriever is not None:
            desc_nodes = [
                item.node for item in self.desc_retriever.retrieve(query)
            ]
        else:
            desc_nodes = []

        if self.exemplar_retriever is not None:
            exemplar_nodes = self.exemplar_retriever.retrieve(query)
            original_size = len(exemplar_nodes)

            slot_nodes = []
            exemplar_nodes = [
                item.node for item in merge_nodes(exemplar_nodes, slot_nodes)
            ][0:original_size]
        else:
            exemplar_nodes = []

        # TODO: Figure out how to better use expectations filter the result set.

        # So we do not have too many exemplars from the same skill
        exemplar_nodes = dedup_nodes(exemplar_nodes, True, self.arity)

        all_nodes = dedup_nodes(desc_nodes + exemplar_nodes, False, 1)

        owners = [
            FrameId(name=item.metadata["owner"])
            for item in all_nodes
        ]

        # Need to remove the bad owner/func/skill/intent.
        skills = [self.module.get_skill(owner) for owner in owners if self.module.has_skill(owner)]
        return skills, exemplar_nodes


def load_context_retrievers(module: Schema, path: str):
    return ContextRetriever(
        module,
        EmbeddingRetriever.load_retriever(path, "desc", RauConfig.get().desc_retrieve_topk),
        HybridRetriever.load_retriever(path, "exemplar", RauConfig.get().exemplar_retrieve_topk),
    )