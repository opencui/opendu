from typing import Any, List

import gin
from FlagEmbedding import LLMEmbedder
from llama_index.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer


embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_instruction = "Represent this sentence in term of implied action:"


def get_embedding() -> BaseEmbedding:
    #return InstructedEmbeddings(embedding_model_name, embedding_instruction)
    return LLMEmbeddings(use_query_as_text=False)


class InstructedEmbeddings(BaseEmbedding):
    def __init__(
            self,
            model_name: str,
            instruction: str,
            **kwargs: Any,
    ) -> None:
        self._model = SentenceTransformer(model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    def expand(self, query) -> str:
        return f"{self._instruction} {query}"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(self.expand(query), normalize_embeddings=True)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts)
        return embeddings.tolist()


class LLMEmbeddings(BaseEmbedding):
    def __init__(
            self,
            task_name: str = "icl",
            use_query_as_text: bool = False,
            **kwargs: Any,
    ) -> None:
        self._model = LLMEmbedder('BAAI/llm-embedder', use_fp16=True)
        self._instruction = ""
        self.task = task_name
        self.use_query_as_text = use_query_as_text
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "llm_embedder"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode_queries(query, task=self.task)

    def _get_text_embedding(self, text: str) -> List[float]:
        if self.use_query_as_text:
            return self._model.encode_queries(text, task=self.task)
        else:
            return self._model.encode_keys(text, task=self.task)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.use_query_as_text:
            embeddings = self._model.encode_queries(texts, task=self.task)
        else:
            embeddings = self._model.encode_keys(texts, task=self.task)
        return embeddings.tolist()