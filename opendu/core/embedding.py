# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import math
from typing import Any, ClassVar, List
import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from opendu.core.config import RauConfig
from jinja2 import Template


# There are two different retrieval tasks:
# 1. desc, where query is query, and key/text is the deescription.
# 2. exemplar, wehre query is query, and key/text is exemplar.

DESC = "desc"
EXEMPLAR = "exemplar"


# We reuse the underlying embedding when we can.
class EmbeddingStore:
    _models: dict[str, SentenceTransformer] = {}

    @classmethod
    def get_model(cls, model_name):
        if model_name in EmbeddingStore._models:
            return EmbeddingStore._models[model_name]
        else:
            model = SentenceTransformer(model_name, device=RauConfig.get().embedding_device, trust_remote_code=True)
            EmbeddingStore._models[model_name] = model.half()
            return model

    @classmethod
    def get_embedding_by_task(cls, kind):
        # make sure 
        if kind != "desc" and kind != "exemplar":
            raise RuntimeError("We can only handle desc and exemplar")
        
        model_name = RauConfig.get().embedding_model
        model = EmbeddingStore.get_model(RauConfig.get().embedding_model)
        if model_name.startswith("dunzhang"):
            return StellaEmbeddings(model, kind)
        elif model_name.startswith("intfloat"):
            return KalmEmbeddings(model, kind)
        elif model_name.startswith("HIT"):
            return KalmEmbeddings(model, kind)
        else:
            return BaaiEmbeddings(model, kind)
    
    @classmethod
    def for_description(cls) -> BaseEmbedding:
        return EmbeddingStore.get_embedding_by_task(DESC)
    
    @classmethod
    def for_exemplar(cls) -> BaseEmbedding:
        return EmbeddingStore.get_embedding_by_task(EXEMPLAR)


# This embedding is based on Stella.
# This embedding has two different modes: one for query, and one for description.
class StellaEmbeddings(BaseEmbedding):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()
    _query_prompt: dict[str, str] = PrivateAttr()
    _text_prompt: dict[str, str] = PrivateAttr()

    def __init__(self, model: SentenceTransformer, kind: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._query_prompt = {"prompt_name": "s2p_query" } if kind == DESC else {"prompt_name": "s2s_query" }
        self._text_prompt = {} if kind == DESC else {"prompt_name": "s2s_query" }

    @classmethod
    def class_name(cls) -> str:
        return "stella"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query, normalize_embeddings=True, show_progress_bar=False, **self._query_prompt)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True, show_progress_bar=False, **self._text_prompt)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, **self._text_prompt)
        return embeddings.tolist()


# This model support many languages, since it is based on qwen 2.5/0.5b
class InstructedEmbeddings(BaseEmbedding):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()

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
        return self._model.encode(self.expand_for_query(query), normalize_embeddings=True, show_progress_bar=False)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(self.expand_for_content(text), normalize_embeddings=True, show_progress_bar=False)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [self._instructions["key"] + key for key in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


class KalmEmbeddings(InstructedEmbeddings):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()

    # We need different instruction pairs for different use cases.
    prompts: ClassVar[dict[str, dict[str, str]]] = {
        DESC : {
            "query": "Instruct: Given a query, retrieve the related intent.\n Query:",
            "key": ""
        },
        EXEMPLAR : {
            "query": "Instruct: Given a query, retrieve the related intent.\n Query:",
            "key": "Instruct: Given a query, Retrieve the related intent.\n Query:",
        }
    }

    def __init__(
        self,
        model: SentenceTransformer,
        kind: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._instructions = KalmEmbeddings.prompts[kind]


# This is for BAAI
class BaaiEmbeddings(BaseEmbedding):
    _instructions: dict[str, str] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()

    # We need different instruction pairs for different use cases.
    prompts: ClassVar[dict[str, dict[str, str]]] = {
        DESC : {
            "query": "",
            "key": "Represent this sentence for searching relevant passages:"
        },
        EXEMPLAR : {
            "query": "",
            "key": ""
        }
    }

    def __init__(
        self,
        model: SentenceTransformer,
        kind: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._instructions = BaaiEmbeddings.prompts[kind]


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


def println(str):
    print(str + "\n")


def similarity(u0, u1, encoder):
    em0 = encoder.get_query_embedding(u0)
    em1 = encoder.get_text_embedding(u1)
    return np.dot(em0, em1) / math.sqrt(np.dot(em0, em0) * np.dot(em1, em1))


class Comparer:
    def __init__(self, encoder0, encoder1):
        self.encoder0 = encoder0
        self.encoder1 = encoder1

    def __call__(self, u0, t0, mode="exemplar"):
        print(u0)
        print(t0)
        if mode == "desc":
            println("desc: " + str(similarity(u0, t0, self.encoder0)))
        else:
            println("exemplar:" + str(similarity(u0, t0, self.encoder1)))


if __name__ == "__main__":

    compare = Comparer(
        EmbeddingStore.get_embedding_by_task("desc"),
        EmbeddingStore.get_embedding_by_task("exemplar")
    )

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to alex"
    t0 = "okay, i'd like to make a transfer of  < transfer_amount >  from checking to  < recipient_name > ."
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'i also need a bus from  < origin >  for 2.'
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer from checking to alex"
    t0 = "okay, i'd like to make a transfer of  < transfer_amount >  from checking to  < recipient_name > ."
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer from checking to khadija."
    t0 = 'i also need a bus from  < origin >  for 2.'
    compare(u0, t0)

    u0 = "let's transfer 610 dollars to their savings account please."
    t0 = 'you could find me a cab to get there for example'
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'that one works. i would like to buy a bus ticket.'
    compare(u0, t0)

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'help user transfer their money from one account to another.'
    compare(u0, t0, "desc")
    
    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer money from one account to another.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer money.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of < transfer_amount > from checking to < recipient_name >."
    t0 = 'transfer money.'
    compare(u0, t0, "desc")

    u0 = "transfer_amount: 370 dollars\nrecipient_name:  khadija\nokay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer money.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer to live agent.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'transfer user to live agent.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija."
    t0 = 'apply new credit card.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'help user transfer their money from one account to another.'
    compare(u0, t0, "desc")
    
    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'transfer money from one account to another.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'transfer money.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'transfer to live agent.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'transfer user to live agent.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'apply new credit card.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'reserve a rooom.'
    compare(u0, t0, "desc")

    u0 = "okay, i'd like to make a transfer of 370 dollars from checking to khadija, and book me a conference room."
    t0 = 'reserve a conference room.'
    compare(u0, t0, "desc")


    u0 = "okay, i'd like the first one."
    t0 = 'I like the < reference > one.'
    compare(u0, t0)

    u0 = "okay, i'd like the first one."
    t0 = 'the < index > one.'
    compare(u0, t0)

    u0 = "< ordinal >: first\nokay, i'd like the first one."
    t0 = 'the < ordinal > one.'
    compare(u0, t0)

    u0 = "change to drink."
    t0 = 'change to < newValue >.'
    compare(u0, t0)

    u0 = "newValue : drink\noldValue : drink\nchange to drink."
    t0 = 'change to < newValue >.'
    compare(u0, t0)