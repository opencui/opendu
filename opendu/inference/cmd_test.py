# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

import getopt
import sys
import json

from opendu.inference.parser import load_parser

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hs:")
    cmd = False
    for opt, arg in opts:
        if opt == "-h":
            print("cmd.py -s <api_schema_directories>")
            sys.exit()
        elif opt == "-s":
            module_paths = arg

    index_path = f"{module_paths}/index"
    # First load the schema info.
    converter = load_parser(module_paths, index_path)

    text = ""

    # Start a loop that will run until the user enters 'quit'.
    while text != "quit":
        # Ask the user for a name.
        text = input('Input your sentence, or enter {"q":question, "u": utterance} or quit:\n')
        # This is how you convert the natural language text into structured representation.

        text = text.strip()

        if text[0] != "{":
            result = converter.understand(text)
        else:
            input = json.loads(text)
            result = converter.decide(input["q"], input["u"])
        print(result)
