import json
import logging
from collections import defaultdict

from opencui.core.annotation import CamelToSnake, ExactMatcher
from opencui.core.embedding import EmbeddingStore
from opencui.core.retriever import (build_nodes_from_skills, create_index, load_context_retrievers)
from opencui.finetune.commons import build_nodes_from_dataset, JsonDatasetFactory
from opencui.inference.converter import Converter
import sys
#
# Converter is a lower level component of inference. This directly use the model.
# This assumes there are fine-tune model already, but use the same client code (albeit different code path)
#
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    factories = [JsonDatasetFactory("./dugsets/sgd/", "sgd")]

    # For now, just use the fix path.
    output = "./test"

    if len(sys.argv) == 1:
        print("usage: python3 opencui/finetune/test.pyt test/train")
        exit()
    else:
        tag = sys.argv[1]

    # Save the things to disk first.
    desc_nodes = []
    exemplar_nodes = []

    factory = JsonDatasetFactory("./dugsets/sgd/", "sgd")

    build_nodes_from_skills(factory.tag, factory.schema.skills, desc_nodes)
    build_nodes_from_dataset(factory.tag, factory[tag], exemplar_nodes)

    # For inference, we only create one index.
    create_index(
        f"{output}/index", "exemplar", exemplar_nodes, EmbeddingStore.for_exemplar()
    )
    create_index(
        f"{output}/index", "desc", desc_nodes, EmbeddingStore.for_description()
    )

    context_retriever = load_context_retrievers(factory.schema, f"{output}/index")

    converter = Converter(context_retriever)

    counts = {
        "exemplar": [0, 0, 0, 0],
        "desc": [0, 0, 0, 0],
        "skill": [0, 0, 0, 0],
        "skills": defaultdict(lambda: [0, 0])
    }

    mode_counts = [0, 0]
    max = -1
    for factory in factories:
        dataset = factory[tag]
        for item in dataset:
            # We only support snake function name.
            owner = CamelToSnake.encode(item["owner"])
            arguments = item["arguments"]
            owner_mode = item["owner_mode"]
            if ExactMatcher.is_good_mode(owner_mode):
                mode_counts[1] += 1
            else:
                mode_counts[0] += 1
            converter.skill_converter.grade(item["utterance"], owner, owner_mode, counts)

    print(json.dumps(counts))
    print(mode_counts)
