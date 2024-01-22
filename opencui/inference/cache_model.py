

from sentence_transformers import SentenceTransformer

from opencui.core.config import LugConfig
from opencui.inference.converter import Generator


if __name__ == "__main__":
    embedder = SentenceTransformer(LugConfig.get().embedding_model, device=LugConfig.get().embedding_device)
    generator = Generator.build()