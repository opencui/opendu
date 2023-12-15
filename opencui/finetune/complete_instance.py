import json
import sys

from opencui import AnnotatedExemplar

if __name__ == "__main__":
    path = sys.argv[1]

    with open(path) as file:
        texts = file.readlines()
        for text in texts:
            jsono = json.loads(text)
            if "flag" not in jsono:
                jsono['flag'] = None
            print(json.dumps(jsono))

