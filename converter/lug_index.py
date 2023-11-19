#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import getopt
import shutil
import logging

from converter.schema_parser import load_all_from_directory
from core.annotation import build_nodes_from_exemplar_store
from core.retriever import create_index, build_nodes_from_skills

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# python lug-index path_for_store_index module_specs_paths_intr
if __name__ == "__main__":
    argv = sys.argv[1:]
    input_paths = ''
    output_path = './index/'
    opts, args = getopt.getopt(argv, "hi:o:")
    for opt, arg in opts:
        if opt == '-h':
            print('lug_index.py -o <output_directory> -i <input_files>')
            sys.exit()
        elif opt == "-i":
            input_paths = arg
        elif opt == "-o":
            outputfile = arg

    modules = input_paths.split(",")

    # For now, we only support single module
    if len(modules) != 1:
        print('lug_index.py -o <output_directory> -i <input_files>')

    try:
        # We assume that there are schema.json, exemplars.json and recognizers.json under the directory
        desc_nodes = []
        exemplar_nodes = []
        for module in modules:
            module_schema, examplers, recognizers = load_all_from_directory(module)
            build_nodes_from_skills(module, module_schema.skills, desc_nodes)
            build_nodes_from_exemplar_store(module, examplers, exemplar_nodes)

        create_index(output_path, "exemplar", exemplar_nodes)
        create_index(output_path, "desc", desc_nodes)
    except Exception as e:
        print(str(e))
        shutil.rmtree(output_path, ignore_errors=True)
