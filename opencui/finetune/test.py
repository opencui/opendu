import logging

from opencui.core.annotation import CamelToSnake
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

    factories = [JsonDatasetFactory("./datasets/sgd/", "sgd")]

    # For now, just use the fix path.
    output = "./test"

    if len(sys.argv) == 1:
        tag = "text"
    else:
        tag = sys.argv[1]

    # Save the things to disk first.
    desc_nodes = []
    exemplar_nodes = []
    tag = "test"
    for factory in factories:
        build_nodes_from_skills(factory.tag, factory.schema.skills, desc_nodes)
        build_nodes_from_dataset(factory.tag, factory[tag], exemplar_nodes)

    # For inference, we only create one index.
    create_index(
        f"{output}/index", "exemplar", exemplar_nodes, EmbeddingStore.for_exemplar()
    )
    create_index(
        f"{output}/index", "desc", desc_nodes, EmbeddingStore.for_description()
    )

    schemas = {factory.tag: factory.schema for factory in factories}

    to_snake = CamelToSnake()
    context_retriever = load_context_retrievers(schemas, f"{output}/index")

    converter = Converter(context_retriever)

    counts = {
        "exemplar": [0, 0, 0, 0],
        "desc": [0, 0, 0, 0],
        "skill": [0, 0]
    }

    total = 0
    max = -1
    for factory in factories:
        dataset = factory[tag]
        for item in dataset:
            if 0 < max < total:
                break
            total += 1
            # We only support snake function name.
            owner = to_snake.encode(item["owner"])
            arguments = item["arguments"]
            owner_mode = item["owner_mode"]
            converter.skill_converter.grade(item["utterance"], owner, owner_mode, counts)

    print(counts)
