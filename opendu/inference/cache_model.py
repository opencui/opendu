# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

from sentence_transformers import SentenceTransformer

from opendu.core.config import RauConfig
from opendu.inference.parser import Decoder

# This script is used to trigger the caching of the models during the docker build to speed up the deployment.
if __name__ == "__main__":
    embedder = SentenceTransformer(
        RauConfig.get().embedding_model,
        device=RauConfig.get().embedding_device,
        trust_remote_code=True)
    generator = Decoder.build()