import logging

from opencui.core.annotation import CamelToSnake, OwnerMode
from opencui.core.embedding import EmbeddingStore
from opencui.core.retriever import (build_nodes_from_skills, create_index, load_context_retrievers)
from opencui.finetune.commons import build_nodes_from_dataset, JsonDatasetFactory
from opencui.inference.converter import Converter

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

    # Save the things to disk first.
    desc_nodes = []
    exemplar_nodes = []
    tag = "train"
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

    counts = [0, 0]
    for factory in factories:
        dataset = factory[tag]
        marker = "### Output:"
        for item in dataset:
            # We only support snake function name.
            target = to_snake.encode(item["owner"])
            arguments = item["arguments"]

            result = converter.understand(item["utterance"])
            if result and result.name == target and OwnerMode[item["owner_mode"]] == OwnerMode.normal:
                counts[1] += 1
            else:
                counts[0] += 1
                print(item["utterance"])
                print(f"{result} != {target} for {item['utterance']} \n")
    print(counts)
