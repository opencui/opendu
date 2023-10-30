#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import getopt
import shutil
import logging

from converter.openai_parser import OpenAIParser
from converter.openapi_parser import OpenAPIParser
from core.annotation import ExemplarStore, SlotRecognizers

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
        for module in modules:
            schema_object = json.load(open(f"{module}/schemas.json"))
            parser = None
            if isinstance(schema_object, list):
                # this is OpenAI schema
                parser = OpenAIParser(schema_object)
            else:
                parser = OpenAPIParser(schema_object)

            examplers = ExemplarStore(**json.load(open(f"{module}/exemplars.json")))
            recognizers = SlotRecognizers(**json.load(open(f"{module}/recognizers.json")))

        print(examplers)
        print(recognizers)
    except Exception as e:
        print(str(e))
        shutil.rmtree(output_path, ignore_errors=True)
