import getopt
import sys
import json

from opencui.inference.converter import load_converter

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:s:")
    cmd = False
    for opt, arg in opts:
        if opt == "-h":
            print("cmd.py -s <api_directories> -i <index_directory>")
            sys.exit()
        elif opt == "-i":
            index_path = arg
        elif opt == "-s":
            module_paths = arg

    # First load the schema info.
    converter = load_converter(module_paths, index_path)

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
