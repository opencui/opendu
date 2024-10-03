# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

from sentence_transformers import SentenceTransformer

from opendu.core.config import RauConfig
from opendu.inference.parser import Generator

# This script is used to trigger the caching of the models during the docker build to speed up the deployment.
if __name__ == "__main__":
    embedder = SentenceTransformer(
        RauConfig.get().embedding_model,
        device=RauConfig.get().embedding_device,
        trust_remote_code=True)
    generator = Generator.build()