import getopt
import sys

from opencui_lug.core.retriever import load_context_retrievers
from opencui_lug.inference.converter import Converter, load_converter
from opencui_lug.inference.schema_parser import load_all_from_directory

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    for opt, arg in opts:
        if opt == "-h":
            print("cmd.py -s <module_directory> -i <index_directory>")
            sys.exit()
        elif opt == "-i":
            index_path = arg
        elif opt == "-s":
            module_path = arg

    # First load the schema info.
    converter = load_converter(module_path, index_path)

    text = ""

    # Start a loop that will run until the user enters 'quit'.
    while text != "quit":
        # Ask the user for a name.
        text = input("Input your sentence, or enter 'quit': ")
        # This is how you convert the natural language text into structured representation.
        result = converter.understand(text)
        print(result)
