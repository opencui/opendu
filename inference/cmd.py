import getopt
import sys

from core.retriever import load_context_retrievers
from inference.converter import Converter
from inference.schema_parser import load_all_from_directory

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    for opt, arg in opts:
        if opt == '-h':
            print('cmd.py -s <module_directory> -i <index_directory>')
            sys.exit()
        elif opt == "-i":
            index_path = arg
        elif opt == "-s":
            module_path = arg

    module_schema, examplers, recognizers = load_all_from_directory(module_path)

    print(module_schema)

    context_retriever = load_context_retrievers({module_path: module_schema}, index_path)
    converter = Converter(context_retriever)

    text = ''

    # Start a loop that will run until the user enters 'quit'.
    while text != 'quit':
        # Ask the user for a name.
        text = input("Input your sentence, or enter 'quit': ")
        result = converter.understand(text)
        print(result)