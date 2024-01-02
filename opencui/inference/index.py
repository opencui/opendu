#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import getopt
import json
import logging
import os.path
import shutil
import sys
import traceback

from opencui.core.annotation import (Exemplar, FrameSchema, build_nodes_from_exemplar_store)
from opencui.core.embedding import EmbeddingStore
from opencui.core.retriever import (build_nodes_from_skills, create_index)
from opencui.inference.schema_parser import load_all_from_directory

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_skill_infos(skills, nodes) -> list[FrameSchema]:
    funcset = {item.node.meta["owner"] for item in nodes}
    return [skills[func] for func in funcset]


def get_exemplars(nodes) -> list[Exemplar]:
    return [Exemplar(owner=item.node.meta["owner"]) for item in nodes]


# python lug-index path_for_store_index module_specs_paths_intr
if __name__ == "__main__":
    argv = sys.argv[1:]
    input_paths = ''
    output_path = './index/'
    opts, args = getopt.getopt(argv, "hi:o:")

    for opt, arg in opts:
        if opt == '-h':
            print('index.py -o <output_directory> -i <input_directory>')
            sys.exit()
        elif opt == "-i":
            input_paths = arg
        elif opt == "-o":
            output_path = arg

    modules = input_paths.split(",")

    # For now, we only support single module
    try:
        # We assume that there are schema.json, exemplars.json and recognizers.json under the directory
        desc_nodes = []
        exemplar_nodes = []
        schemas = {}
        for module in modules:
            print(f"load {module}")
            module_schema, examplers, recognizers = load_all_from_directory(module)
            print(module_schema)
            build_nodes_from_skills(module, module_schema.skills, desc_nodes)
            build_nodes_from_exemplar_store(module, examplers, exemplar_nodes)
            schemas[module] = module_schema

        # now we create index for both desc and exemplar for all modules.
        create_index(output_path, "exemplar", exemplar_nodes,
                     EmbeddingStore.for_exemplar())
        create_index(output_path, "desc", desc_nodes,
                     EmbeddingStore.for_description())
    except:
        traceback.print_exc()
        shutil.rmtree(output_path, ignore_errors=True)
