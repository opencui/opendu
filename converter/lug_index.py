#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import re, sys, getopt
import shutil
import logging

from urllib.parse import urlparse

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# python lug-index path_for_store_index module_specs_paths_intr
if __name__ == "__main__":
    argv = sys.argv[1:]
    inputfiles = ''
    output_path = './index/'
    opts, args = getopt.getopt(argv, "hi:o:")
    for opt, arg in opts:
        if opt == '-h':
            print('lug_index.py -o <output_directory> -i <input_files>')
            sys.exit()
        elif opt == "-i":
            inputfiles = arg
        elif opt == "-o":
            outputfile = arg

    files = inputfiles.split(",")
    if len(files)%3 != 0:
        print("Expecting input files are paired in specs, exemplars, and recognizer")
        sys.exit()

    try:

        parser = None
    except Exception as e:
        print(str(e))
        shutil.rmtree(output_path, ignore_errors=True)
