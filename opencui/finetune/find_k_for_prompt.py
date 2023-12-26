import logging

from datasets import Dataset
import numpy as np
from opencui.core.config import LugConfig
from opencui.core.embedding import EmbeddingStore
from opencui.core.retriever import (ContextRetriever, build_desc_index, load_context_retrievers)
from opencui.finetune.commons import build_dataset_index, JsonDatasetFactory


def compute_k(dataset: Dataset, retrieve: ContextRetriever):
    counts = [0, 0]
    for item in dataset:
        skills, exemplars = retrieve(item["utterance"])
        if item["owner"] == "NONE":
            continue

        intents = set([skill["name"] for skill in skills])
        counts[0] += 1
        if item["owner"] in intents:
            counts[1] += 1
        else:
            print(f">>>>>{item}: \n +++++ {skills}\n --------{exemplars} \n\n")

    return counts


def compute_k_examplar(dataset: Dataset, retrieve: ContextRetriever):
    first_indexes = []
    first_scores = []
    for item in dataset:
        results = retrieve.retrieve_by_exemplar(item["utterance"])
        if item["owner"] == "NONE":
            continue
        gindex = 100
        gscore = 0
        for index, result in enumerate(results):
            if result.node.metadata["owner"] == item["owner"]:
                gindex = index
                gscore = result.score
        first_indexes.append(gindex)
        first_scores.append(gscore)
    return first_indexes, first_scores


def find_percentile(a, percentile=98, reverse=False):
    a.sort()
    print(a)
    npa = np.array(a)
    return np.percentile(npa, percentile)


#
# It is really import that we get the hyperparameter right. For fine-tune the generator in the RAG,
# we need to make sure the prompt template can be instantiated to meet the certain criteria.
# In particular, we need to find some constant in terms of how many function and exemplars do we need
# include to have a high enough probability to have correct function included in the context.
#
if __name__ == "__main__":
    # The first thing is to create the schema and create the datasets of annotated exemplars.
    # Then create the index for both descriptions and exemplars on training split.
    # Then define the prompt.
    # Then figure out the good k using validation split. These Ks will be used for inference and training.
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    LugConfig.embedding_device = "cuda:0"

    factories = [JsonDatasetFactory("./datasets/sgd", "sgd")]

    # For now, just use the fix path.
    output = "./output"

    print("building index first.")
    for factory in factories:
        build_desc_index(factory.tag, factory.schema,
                         f"{output}/index/{factory.tag}",
                         EmbeddingStore.for_description())
        build_dataset_index(factory.tag, factory["train"],
                            f"{output}/index/{factory.tag}",
                            EmbeddingStore.for_exemplar())

    retrievers = []
    for factory in factories:
        retrievers.append(
            load_context_retrievers({factory.tag: factory.schema},
                                    f"{output}/index/{factory.tag}"))

    for index in range(len(factories)):
        factory = factories[index]
        searcher = retrievers[index]
        ds = factory["train"]
        print(compute_k(ds, searcher))
        first_indexes, first_scores = compute_k_examplar(ds, searcher)
        print(find_percentile(first_indexes, 99))
        print(find_percentile(first_scores, 1))

