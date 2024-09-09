# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

from sentence_transformers import SentenceTransformer

from opencui.core.config import LugConfig
from opencui.inference.converter import Generator

# This script is used to trigger the caching of the models during the docker build to speed up the deployment.
if __name__ == "__main__":
    embedder = SentenceTransformer(LugConfig.get().embedding_model, device=LugConfig.get().embedding_device)
    generator = Generator.build()