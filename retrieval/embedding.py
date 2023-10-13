from typing import Any, List

import gin
from llama_index.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer


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


@gin.configurable
def get_embedding(model_name: str, instruction: str):
    return InstructedEmbeddings(model_name=model_name, instruction=instruction)