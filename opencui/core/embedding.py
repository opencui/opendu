import math
from typing import Any, List

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

from opencui.core.config import LugConfig


# We reuse the underlying embedding when we can.
class EmbeddingStore:
    _models: dict[str, SentenceTransformer] = {}

    # We need different instruction pairs for different use cases.
    INSTRUCTIONS = {
        "qa": {
            "query":
            "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "icl": {
            "query":
            "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "desc": {
            "query":
            "Convert this text into vector to look for useful function: ",
            "key":
            "Convert this function description into vector for retrieval: ",
        },
        "exemplar": {
            "query":
            "Convert this text into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "chat": {
            "query":
            "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query":
            "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "tool": {
            "query":
            "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
        "convsearch": {
            "query":
            "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
        "baai_desc": {
            "query": "",
            "key": "Represent this sentence for searching relevant passages:"
        },
        "baai_exemplars": {
            "query": "",
            "key": ""
        }
    }

    @classmethod
    def get_model(cls, model_name):
        if model_name in EmbeddingStore._models:
            return EmbeddingStore._models[model_name]
        else:
            model = SentenceTransformer(model_name, device=LugConfig.get().embedding_device)
            EmbeddingStore._models[model_name] = model
            return model

    @classmethod
    def get_embedding_by_task(cls, kind):
        model = EmbeddingStore.get_model(LugConfig.get().embedding_model)
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])

    @classmethod
    def for_description(cls) -> BaseEmbedding:
        model = EmbeddingStore.get_model(LugConfig.get().embedding_model)
        kind = LugConfig.get().embedding_desc_prompt
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])

    @classmethod
    def for_exemplar(cls) -> BaseEmbedding:
        model = EmbeddingStore.get_model(LugConfig.get().embedding_model)
        kind = LugConfig.get().embedding_desc_prompt
        return InstructedEmbeddings(model, EmbeddingStore.INSTRUCTIONS[kind])


class InstructedEmbeddings(BaseEmbedding):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()

    def __init__(
        self,
        model: SentenceTransformer,
        instruction: dict[str, str],
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._instructions = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    # embedding might have two different modes: one for query, and one for key/text.

    def expand_for_content(self, query):
        return f"{self._instructions['key']} {query}"

    def expand_for_query(self, query):
        return f"{self._instructions['query']} {query}"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(self.expand_for_query(query), normalize_embeddings=True)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(self.expand_for_content(text), normalize_embeddings=True)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [self._instructions["key"] + key for key in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

def similarity(u0, u1, encoder):
    em0 = encoder.get_query_embedding(u0)
    em1 = encoder.get_text_embedding(u1)
    return np.dot(em0, em1) / math.sqrt(np.dot(em0, em0) * np.dot(em1, em1))


class Comparer:
    def __init__(self, encoder0, encoder1):
        self.encoder0 = encoder0
        self.encoder1 = encoder1

    def __call__(self, u0, t0):
        print(similarity(u0, t0, self.encoder0))
        print(similarity(u0, t0, self.encoder1))


if __name__ == "__main__":

    compare = Comparer(
        EmbeddingStore.get_embedding_by_task("desc"),
        EmbeddingStore.get_embedding_by_task("exemplar")
    )

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija"
    t0 = "okay, i'd like to make a transfer of  < transfer_amount >  from checking to  < recipient_name > ."

    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'i also need a bus from  < origin >  for 2.'
    compare(u0, t0)

    u0 = "let's transfer 610 dollars to their savings account please."
    t0 = 'you could find me a cab to get there for example'
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'that one works. i would like to buy a bus ticket.'
    compare(u0, t0)


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer money.'
    compare(u0, t0)