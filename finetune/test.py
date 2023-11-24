import logging

from inference.converter import Converter
from core.embedding import EmbeddingStore
from core.retriever import load_context_retrievers, build_nodes_from_skills, create_index
from finetune.commons import build_nodes_from_dataset

#
# Converter is a lower level component of inference. This directly use the model.
# This assumes there are fine-tune model already, but use the same client code (albeit different code path)
#
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    from finetune.sgd import SGD
    from core.config import LugConfig

    LugConfig.embedding_device = "cuda"
    factories = [
        SGD("/home/sean/src/dstc8-schema-guided-dialogue/")]

    # For now, just use the fix path.
    output = "./test"

    # Save the things to disk first.
    desc_nodes = []
    exemplar_nodes = []
    for factory in factories:
        build_nodes_from_skills(factory.tag, factory.schema.skills, desc_nodes)
        build_nodes_from_dataset(factory.tag, factory.build("train"), exemplar_nodes)

    # For inference, we only create one index.
    create_index(f"{output}/index", "exemplar", exemplar_nodes, EmbeddingStore.for_exemplar())
    create_index(f"{output}/index", "desc", desc_nodes, EmbeddingStore.for_description())

    schemas = {factory.tag: factory.schema for factory in factories}

    print(schemas)

    context_retriever = load_context_retrievers(schemas, f"{output}/index")

    converter = Converter(context_retriever)

    counts = [0, 0]
    for factory in factories:
        dataset = factory.build("test")
        marker = "### Output:"
        for item in dataset:
            result = converter.understand(item["utterance"])
            target = item['owner']
            arguments = item["arguments"]
            if result and result.name == target:
                counts[1] += 1
            else:
                print(f"{result} != {target}\n")
    print(counts)
