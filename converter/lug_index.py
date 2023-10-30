#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import re
import sys
import gin
import shutil
import logging

from urllib.parse import urlparse

from llama_index import ServiceContext, StorageContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader, SimpleKeywordTableIndex
from llama_index import set_global_service_context


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# python fc-index index_persist_path collection_path...
# collection_path
#     data/
#     /data
#     https://abc.com/xyz.md
#     https://<token>@github.com/<org>/<repo>
#     https://<token>@github.com/<org>/<repo>/tree/<tag_name|branch_name>/<sub_dir>
#     https://<token>@github.com/<org>/<repo>/blob/<tag_name|branch_name|commit_id>/<sub_dir>/<file_name>.md
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(0)

    # We assume that there output directory is the first argument, and the rest is input directory
    output = sys.argv[1]
    gin.parse_config_file('index.gin')

    # init download hugging fact model
    service_context = ServiceContext.from_defaults(
        llm=None,
        llm_predictor=None,
        embed_model=get_embedding(),
    )

    storage_context = StorageContext.from_defaults()

    set_global_service_context(service_context)

    documents = []
    for file_path in sys.argv[2:]:
        if os.path.isfile(file_path) and file_path.endswith(".md"):
            print(map_func["file"])
            documents.extend(map_func["file"](file_path))
        elif os.path.isdir(file_path):
            documents.extend(map_func["dir"](file_path))
        else:
            match_github = re.search(re_github, file_path)
            if match_github:
                documents.extend(map_func["github"](match_github))
                continue

            match_url = urlparse(file_path)
            if match_url.scheme and match_url.netloc:
                documents.extend(map_func["url"](file_path))
                continue

    # exclude these things from considerations.
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["file_name", "content_type"]
        doc.excluded_embed_metadata_keys = ["file_name", "content_type"]

    try:
        embedding_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context)
        keyword_index = SimpleKeywordTableIndex(
            documents, storage_context=storage_context)

        embedding_index.set_index_id("embedding")
        embedding_index.storage_context.persist(persist_dir=output)
        keyword_index.set_index_id("keyword")
        keyword_index.storage_context.persist(persist_dir=output)
    except Exception as e:
        print(str(e))
        shutil.rmtree(output, ignore_errors=True)
