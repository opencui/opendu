from typing import Any, List

from llama_index.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

from llama_index.bridge.pydantic import PrivateAttr

embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_instruction = "Represent this sentence in term of implied action:"

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "iclda": {
        "query": "Convert the dialog acts of this example into vector to look for useful dialog act examples: ",
        "key": "Convert the dialog acts of this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}


def get_embedding(kind: str = 'iclda') -> BaseEmbedding:
    return InstructedEmbeddings('BAAI/llm-embedder', INSTRUCTIONS[kind])


class InstructedEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instructions: dict[str, str] = PrivateAttr()

    def __init__(
            self,
            model_name: str,
            instruction: dict[str, str],
            **kwargs: Any,
    ) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")
        self._instructions = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    # embedding might have two different modes: one for query, and one for key/text.
    def expand(self, query: str, mode: str) -> str:
        return f"{self._instructions[mode]} {query}"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(self.expand(query, 'query'), normalize_embeddings=True)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(self.expand(text, 'key'), normalize_embeddings=True)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [self._instructions["key"] + key for key in texts]
        embeddings = self._model.encode(texts)
        return embeddings.tolist()
